#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <CL/opencl.h>

#define MAXDEVICE 10
#define MAXK 5000000
#define N 1024
#define EDGE 32
#define SWEEP 500
#define TIMES 50
#define NANO2SECOND 1000000000.0
#define AOCL_ALIGNMENT 64

void usage () {
	printf("Usage:\n");
	printf("       ./Ising-opencl [kernel file] [spin configuration]\n");
	exit(0);
}

/*
   	0  1  2  3  4  5  6  7     
   	8  9 10 11 12 13 14 15   
   16 17 18 19 20 21 22 23     s0  s1  s2
   24 25 26 27 28 29 30 31     s3   h  s4
   32 33 34 35 36 37 38 39     s5  s6  s7
   40 41 42 43 44 45 46 47
   48 49 50 51 52 53 54 55
   56 57 58 59 60 61 62 63
 */

int relation (int a, int b) {
	switch (b-a) {
		case 0:
			return 8;
		case -EDGE-1:
			return 0;
		case -EDGE:
			return 1;
		case -EDGE+1:
			return 2;
		case -1:
			return 3;
		case 1:
			return 4;
		case EDGE-1:
			return 5;
		case EDGE:
			return 6;
		case EDGE+1:
			return 7;
		default:
			return -1;
	}
}

unsigned char kernelBuffer[MAXK];

int main (int argc, char *argv[]) {
	if (argc != 3) 
		usage();
	
	srand(123);

	struct timeval timeStart, timeEnd;

	// Platform info
	cl_int status;
	cl_platform_id platform_id[2]; // Nvidia and Intel
	cl_int platform_id_got;
	status = clGetPlatformIDs(2, platform_id, &platform_id_got);
	assert(status == CL_SUCCESS);
	printf("%d platform found\n", platform_id_got);
	for (int i = 0; i < platform_id_got; i++) {
		char buffer[256];
		size_t length;
		clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 256, buffer, &length);
		buffer[length] = '\0';
		printf("    platform name: %s\n", buffer);
	}

	// Device info
	cl_device_id DEVICES[MAXDEVICE];
	cl_int DEVICE_id_got;
	status = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, MAXDEVICE, DEVICES, 
			&DEVICE_id_got); // GPU or CPU
	assert(status == CL_SUCCESS);
	printf("There are %d FPGA devices\n", DEVICE_id_got);
	for (int i = 0; i < DEVICE_id_got; i++) {
		char buffer[256];
		size_t length;
		clGetDeviceInfo(DEVICES[i], CL_DEVICE_NAME, MAXDEVICE, buffer, &length);
		buffer[length] = '\0';
		printf("    Device name: %s\n", buffer);
	}
	
	// Context consists of devices that will participate the computation
	cl_context context = clCreateContext(NULL, 1, DEVICES, NULL, NULL, &status); // 1~3
	assert(status == CL_SUCCESS);

	// Command Queue: send commands to devices
	cl_command_queue commandQueue = clCreateCommandQueue(context, DEVICES[0], 
			CL_QUEUE_PROFILING_ENABLE, &status);
	assert(status == CL_SUCCESS);


	// Kernel Program: we must specify the computation as a kernel
	FILE *kernelfp = fopen(argv[1], "rb");
	assert(kernelfp != NULL);
	const unsigned char *constKernelSource = kernelBuffer;
	size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
	printf("The size of kernel binary is %zu\n", kernelLength);
	cl_program program = clCreateProgramWithBinary(context, 1, DEVICES,
			&kernelLength, &constKernelSource, &status, NULL);
	fclose(kernelfp);
	assert(status == CL_SUCCESS);

	// Build Program
	status = clBuildProgram(program, 1, DEVICES, NULL, NULL, NULL); // 1~3
	assert(status == CL_SUCCESS);
	printf("Build program completes\n");

	// Create Kernel (which function in program)
	cl_kernel kernel = clCreateKernel(program, "sa", &status);
	assert(status == CL_SUCCESS);
	printf("Build kernel completes\n");

	// Prepare problems input
	cl_int* couplings = (cl_int*)malloc(8 * N * sizeof(cl_int));
	posix_memalign((void*)&couplings, AOCL_ALIGNMENT, 8*N*sizeof(cl_int));
	assert(couplings != NULL);
	memset(couplings, '\0', 8*N*sizeof(int));
	cl_int* fields = (cl_int*)malloc(N * sizeof(cl_int));
	posix_memalign((void*)&fields, AOCL_ALIGNMENT, N*sizeof(cl_int));
	assert(fields != NULL);
	memset(fields, '\0', N*sizeof(int));
	cl_float* randomLogT = (cl_float*)malloc(SWEEP * sizeof(cl_float));
	posix_memalign((void*)&randomLogT, AOCL_ALIGNMENT, SWEEP*sizeof(cl_float));
	assert(randomLogT != NULL);
	cl_int* spin_in = (cl_int*)malloc(N * sizeof(cl_int));
	posix_memalign((void*)&spin_in, AOCL_ALIGNMENT, N*sizeof(cl_int));
	assert(spin_in != NULL);
	cl_int* spin_out = (cl_int*)malloc(N * sizeof(cl_int));
	posix_memalign((void*)&spin_out, AOCL_ALIGNMENT, N*sizeof(cl_int));
	assert(spin_out != NULL);
	
	// Read couplings file 
	FILE *instance = fopen(argv[2], "r");
	assert(instance != NULL);
	int a, b, w;
	fscanf(instance, "%d", &a);
	while (!feof(instance)) {
		fscanf(instance, "%d%d%d", &a, &b, &w);
		int r = relation(a, b);
		if (r == -1) {
			assert(-1);
		} else if (r == 8) {
			fields[a] = w;
		} else {
			couplings[8*a+r] = w;
			r = relation(b, a);
			couplings[8*b+r] = w;
		}
	}
	fclose(instance);
	printf("Finish reading instance\n");

	// Create Buffer (Pass data to device buffer)
	cl_mem buffer_couplings = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
			8 * N * sizeof(cl_int), couplings, &status);
	assert(status == CL_SUCCESS);
	cl_mem buffer_fields = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
			N * sizeof(cl_int), fields, &status);
	assert(status == CL_SUCCESS);
	cl_mem buffer_spin_out = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
			N * sizeof(cl_int), spin_out, &status);
	assert(status == CL_SUCCESS);
		
	// Parameter Linking: link allocated buffers to program's (kernel.cl) function's parametes
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_couplings);
	assert(status == CL_SUCCESS);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_fields);
	assert(status == CL_SUCCESS);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_spin_out);
	assert(status == CL_SUCCESS);
		
	// start annealing
	printf("Start Annealing\n");

	// Shape of Data: How to put data in device
	int results[TIMES] = {0};
	float increase = (3.0 - 0.1) / (float)SWEEP;
	gettimeofday(&timeStart, NULL);
	for (int x = 0; x < TIMES; x++) {
		for (int j = 0; j < N; j++)
			spin_in[j] = ((rand()&1)<<1)-1;
		
		// create buffer
		cl_mem buffer_spin_in = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				N * sizeof(cl_int), spin_in, &status);
		assert(status == CL_SUCCESS);
		
		float beta = 0.1;	
		for (int i = 0; i < SWEEP; i++) {
			beta += increase;
			randomLogT[i] = -log(rand() / (float) RAND_MAX) / beta / 2.0;
		}
		cl_mem buffer_randomLogT = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				SWEEP * sizeof(cl_float), randomLogT, &status);
		assert(status == CL_SUCCESS);

		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_randomLogT);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_spin_in);
		assert(status == CL_SUCCESS);

		cl_event event;
		status = clEnqueueTask(commandQueue, kernel, 0, NULL, &event);
		assert(status == CL_SUCCESS);
		//clFinish(commandQueue);

		
		clWaitForEvents(1, &event);
		cl_ulong tQ, tSub, tS, tE;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, 
				sizeof(cl_ulong), &tQ, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, 
				sizeof(cl_ulong), &tSub, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, 
				sizeof(cl_ulong), &tS, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, 
				sizeof(cl_ulong), &tE, NULL);
		printf("Queued time %f (s)\n", (tSub-tQ)/1000000000.0);
		printf("Submission time %f (s)\n", (tS-tSub)/1000000000.0);
		printf("Execution time %f (s)\n", (tE-tS)/1000000000.0);
		

		// Get Result from device
		status = clEnqueueReadBuffer(commandQueue, buffer_spin_out, CL_TRUE, 0, N * sizeof(cl_int),
				spin_out, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		clReleaseMemObject(buffer_randomLogT);
		clReleaseMemObject(buffer_spin_in);
		for (int i = 0; i < N; i++) { 
			results[x] += -spin_out[i] * fields[i];
			for (int j = i+1; j < N; j++) {
				int r = relation(i, j);
				if (r == -1)
					continue;
				results[x] += -spin_out[i] * spin_out[j] * couplings[8*i+r];
			}
		}
	}
	gettimeofday(&timeEnd, NULL);

	long seconds = timeEnd.tv_sec - timeStart.tv_sec;
	long micro = ((seconds * 1000000) + timeEnd.tv_usec) - timeStart.tv_usec;
	float time_statistics = micro / (float) 1000000.0;
	printf("%d times in %lf. Average time: %lf\n", TIMES, time_statistics, time_statistics/(float)TIMES);

	// Write statistics to file
	FILE *output;
	output = fopen("output.txt", "w");
	for (int i = 0; i < TIMES; i++)
 		fprintf(output, "%d\n", results[i]);
	fclose(output);

	// Release Objects
	free(couplings);
	free(fields);
	clReleaseContext(context);
	clReleaseCommandQueue(commandQueue);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(buffer_couplings);
	clReleaseMemObject(buffer_fields);
	clReleaseMemObject(buffer_spin_out);
		
	return 0;
}

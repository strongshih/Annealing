#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <CL/opencl.h>

#define MAXDEVICE 10
#define MAXK 2048
#define N 512
#define TIMES 1024
#define NANO2SECOND 1000000000.0

void usage() {
	printf("Usage:\n");
	printf("       ./Ising-opencl [kernel file] [spin configuration]\n");
	exit(0);
}

int main (int argc, char *argv[]) {
	if (argc != 3) 
		usage();

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
	printf("There are %d GPU devices\n", DEVICE_id_got);
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
	FILE *kernelfp = fopen(argv[1], "r");
	assert(kernelfp != NULL);
	char kernelBuffer[MAXK];
	const char *constKernelSource = kernelBuffer;
	size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
	printf("The size of kernel source is %zu\n", kernelLength);
	cl_program program = clCreateProgramWithSource(context, 1, 
			&constKernelSource, &kernelLength, &status);
	fclose(kernelfp);
	assert(status == CL_SUCCESS);

	// Build Program
	status = clBuildProgram(program, 1, DEVICES, NULL, NULL, NULL); // 1~3
	assert(status == CL_SUCCESS);
	printf("Build program completes\n");

	// Create Kernel (which function in program)
	cl_kernel kernel = clCreateKernel(program, "ising", &status);
	assert(status == CL_SUCCESS);
	printf("Build kernel completes\n");

	// Prepare problems input
	cl_int* couplings = (cl_int*)malloc(N * N * sizeof(cl_int));
	cl_int* results = (cl_int*)malloc(TIMES * sizeof(cl_int));
	assert(results != NULL && couplings != NULL);
	memset(couplings, '\0', N*N*sizeof(int));

	// Read couplings file 
	FILE *instance = fopen(argv[2], "r");
	assert(instance != NULL);
	int a, b, w;
	fscanf(instance, "%d", &a);
	while (!feof(instance)) {
		fscanf(instance, "%d%d%d", &a, &b, &w);
		couplings[a * N + b] = w;
		couplings[b * N + a] = w;
	}
	fclose(instance);
	printf("Finish reading instance\n");

	// Create Buffer (Pass data to device buffer)
	cl_mem buffer_couplings = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
			N * N * sizeof(cl_int), couplings, &status);
	assert(status == CL_SUCCESS);
	cl_mem buffer_results = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
			TIMES * sizeof(cl_int), results, &status);
	assert(status == CL_SUCCESS);
	// printf("Build buffers completes\n");
		
	// Parameter Linking: link allocated buffers to program's (kernel.cl) function's parametes
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_couplings);
	assert(status == CL_SUCCESS);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_results);
	assert(status == CL_SUCCESS);
	// printf("Set kernel arguments completes\n");

	// start annealing
	printf("Start Annealing\n");

	// Shape of Data: How to put data in device
	size_t globalThreads[] = {TIMES};
	size_t localThreads[] = {1};
	cl_event events[1];
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalThreads, localThreads,
			0, NULL, &(events[0]));
	assert(status == CL_SUCCESS);
	status = clWaitForEvents(1, &(events[0]));
	assert(status == CL_SUCCESS);
	// printf("Specify the shape of the domain completes.\n");

	// Get Result from device
	status = clEnqueueReadBuffer(commandQueue, buffer_results, CL_TRUE, 0, TIMES * sizeof(cl_int),
			results, 0, NULL, NULL);
	assert(status == CL_SUCCESS);
	// printf("Kernel execution completes\n");

	// calculate annealing time
	cl_ulong timeStart, timeEnd;
	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START,
			sizeof(cl_ulong), &timeStart, NULL);
	assert(status == CL_SUCCESS);
	status = clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_END,
			sizeof(cl_ulong), &timeEnd, NULL);
	assert(status == CL_SUCCESS);

	float time_statistics = (timeEnd - timeStart) / NANO2SECOND;
	printf("%d times in %lf. Average time: %lf\n", TIMES, time_statistics, time_statistics/TIMES);

	// Write statistics to file
	FILE *output;
	output = fopen("output.txt", "w");
	for (int i = 0; i < TIMES; i++)
 		fprintf(output, "%d\n", results[i]);
	fclose(output);

	// Release Objects
	free(results);
	free(couplings);
	clReleaseContext(context);
	clReleaseCommandQueue(commandQueue);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(buffer_results);
	clReleaseMemObject(buffer_couplings);
	return 0;
}

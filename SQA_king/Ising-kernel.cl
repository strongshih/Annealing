#define N 1024
#define N_4 256
#define EDGE 32
#define EDGE_SHIFT 5
#define EDGE_2 16
#define EDGE_SHIFT_2 4
#define BITMASK 0x1f
#define BITMASK_2 0xf
#define SWEEP 500
#define M 4

__kernel void sqa(__global int *restrict couplings,
				  __global float *restrict randomLog,
				  __global float *restrict Jtrans,
				  __global int *restrict SPIN_IN, 
				  __global int *restrict SPIN_OUT,
				  __global int *restrict fields)
{
	__private float lrandom[M];
	__private int localJ[N][8];
	__private int lspin[M][N];
	__private int lh[N];
	__private float lJtrans;

	lJtrans = Jtrans[0];

	// copy global memory data to the internal memory;
	for (int j = 0; j < N; j++)
		# pragma unroll 2
		for (int i = 0; i < M; i++)
			lspin[i][j] = SPIN_IN[i*N+j];
	
	// copy random
	#pragma unroll
	for (int i = 0; i < M; i++)
		lrandom[i] = randomLog[i];

	// copy external fields
	for (int i = 0; i < N; i++)
		lh[i] = fields[i];

	#pragma unroll 4
    for (int i = 0; i < N; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            localJ[i][j] = couplings[8*i+j];
        }
	
	for (int i = 0; i < M; i++)
        lrandom[i] = randomLog[i];

	for (int m = 0; m < M; m++) {
		float currentR = lrandom[m];
		int firstUnit[4] = {0, 1, EDGE, EDGE+1};

		for (int s = 0; s < 4; s++) {
			int currentUnit = firstUnit[s];
			#pragma ivdep array(lspin)
			#pragma unroll 2
			for (int n = 0; n < N_4; n++) {
				int row = n>>(EDGE_SHIFT_2);
				int col = n&(BITMASK_2);
				int currentSpin = (currentUnit + ((row<<1)<<EDGE_SHIFT) + (col<<1));
				int diff = 0;

				int edge = currentSpin-EDGE-1;
				int s0 = ((currentSpin&BITMASK) == 0 || currentSpin < EDGE) ?
							0 : lspin[m][edge]*localJ[currentSpin][0];  

				edge = currentSpin-EDGE;
				int s1 = (currentSpin < EDGE) ?
							0 : lspin[m][edge]*localJ[currentSpin][1];

				edge = currentSpin-EDGE+1;
				int s2 = (((currentSpin+1)&BITMASK) == 0 || currentSpin < EDGE) ?
							0 : lspin[m][edge]*localJ[currentSpin][2];

				edge = currentSpin-1;
				int s3 = (currentSpin&BITMASK) == 0 ?
							0 : lspin[m][edge]*localJ[currentSpin][3];

				edge = currentSpin+1;
				int s4 = ((currentSpin+1)&BITMASK) == 0 ?
							0 : lspin[m][edge]*localJ[currentSpin][4];
				int firstUnit[4] = {0, 1, EDGE, EDGE+1};

				edge = currentSpin+EDGE-1;
				int s5 = ((currentSpin&BITMASK) == 0 || currentSpin >= EDGE*(EDGE-1)) ?
							0 : lspin[m][edge]*localJ[currentSpin][5];

				edge = currentSpin+EDGE;
				int s6 = (currentSpin >= EDGE*(EDGE-1)) ?
							0 : lspin[m][edge]*localJ[currentSpin][6];

				edge = currentSpin+EDGE+1;
				int s7 = (((currentSpin+1)&BITMASK) == 0 || currentSpin >= EDGE*(EDGE-1)) ?
							0 : lspin[m][edge]*localJ[currentSpin][7];
				diff = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + lh[currentSpin];

				unsigned int up = (m!=0) ? m-1: M-1;
				unsigned int down = (m!=M-1) ? m+1: 0;
				diff -= lJtrans*M*(lspin[up][currentSpin] + lspin[down][currentSpin]);
				diff *= lspin[m][currentSpin];

				if (diff < currentR)
					lspin[m][currentSpin] *= -1;
			}
		}
	}

	// copy spin values from the internal memory to the global memory
	for (int j = 0; j < N; j++)
		# pragma unroll
		for (int i = 0; i < M; i++)
			SPIN_OUT[i*N+j] = lspin[i][j];
}

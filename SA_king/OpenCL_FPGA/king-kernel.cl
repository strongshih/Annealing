#define N 16384
#define N_4 4096
#define EDGE 128
#define EDGE_SHIFT 7
#define EDGE_2 64
#define EDGE_SHIFT_2 6
#define BITMASK 0x7f
#define BITMASK_2 0x3f
#define SWEEP 500

__kernel void sa(__global int *restrict couplings,
				  __global int *restrict fields,
				  __global float *restrict randomLog,
				  __global int *restrict SPIN_IN, 
				  __global int *restrict SPIN_OUT)
{
	__private float lrandom[SWEEP];
	__private int localJ[N][8];
	__private int lh[N];
	__private int lspin[N];

	// copy global memory data to the internal memory;
	#pragma unroll 4
	for (int i = 0; i < N; i++)
		#pragma unroll
		for (int j = 0; j < 8; j++) {
			localJ[i][j] = couplings[8*i+j];
		}
	for (int i = 0; i < N; i++)
		lspin[i] = SPIN_IN[i];
	for (int i = 0; i < N; i++)
		lh[i] = fields[i];
	for (int i = 0; i < SWEEP; i++)
		lrandom[i] = randomLog[i];

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
	int firstUnit[4] = {0, 1, EDGE, EDGE+1};
	for (int t = 0; t < SWEEP; t++) {
		float currnetR = lrandom[t];
		for (int s = 0; s < 4; s++) {
			int currentUnit = firstUnit[s];
			#pragma ivdep array(lspin)
			for (int n = 0; n < N_4; n++) {
				int row = n>>(EDGE_SHIFT_2);
				int col = n&(BITMASK_2);
				int currentSpin = (currentUnit + ((row<<EDGE_SHIFT)<<1) + (col<<1));
				int energy = 0;
				
				int edge = currentSpin-EDGE-1;
				int s0 = ((currentSpin&BITMASK) == 0 || currentSpin < EDGE) ? 
							0 : lspin[edge]*localJ[currentSpin][0];  
				
				edge = currentSpin-EDGE;
				int s1 = (currentSpin < EDGE) ? 
							0 : lspin[edge]*localJ[currentSpin][1];
				
				edge = currentSpin-EDGE+1;
				int s2 = (((currentSpin+1)&BITMASK) == 0 || currentSpin < EDGE) ? 
							0 : lspin[edge]*localJ[currentSpin][2];
				
				edge = currentSpin-1;
				int s3 = (currentSpin&BITMASK) == 0 ? 
							0 : lspin[edge]*localJ[currentSpin][3];

				edge = currentSpin+1;
				int s4 = ((currentSpin+1)&BITMASK) == 0 ? 
							0 : lspin[edge]*localJ[currentSpin][4];

				edge = currentSpin+EDGE-1;
				int s5 = ((currentSpin&BITMASK) == 0 || currentSpin >= EDGE*(EDGE-1)) ? 
							0 : lspin[edge]*localJ[currentSpin][5];

				edge = currentSpin+EDGE;
				int s6 = (currentSpin >= EDGE*(EDGE-1)) ? 
							0 : lspin[edge]*localJ[currentSpin][6];

				edge = currentSpin+EDGE+1;
				int s7 = (((currentSpin+1)&BITMASK) == 0 || currentSpin >= EDGE*(EDGE-1)) ? 
							0 : lspin[edge]*localJ[currentSpin][7];
				energy = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + lh[currentSpin];
				energy *= lspin[currentSpin];
				
				if (energy < currnetR)
					lspin[currentSpin] *= -1;
			}
		}
	}
	#pragma unroll 4
	for (int i = 0; i < N; i++)
		SPIN_OUT[i] = lspin[i];
}

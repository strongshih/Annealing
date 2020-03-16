#define N 512
#define SWEEP 200
#define TIMES 1024
#define MAX 4294967295

uint xorshift32(uint *state)
{
	uint x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

__kernel void ising(__global const int* restrict couplings, 
					__global int* restrict results)
{
	// random number
	uint randnum = 1337;
	float beta = 0.1; // from 0.1 to 3.0
	float increase = (3.0 - 0.1) / (float) SWEEP;

	// spin initialization
	__local int spins[N];
	__local int difference[N];
	__local int schedule[SWEEP];

	for (int t = 0; t < TIMES; t++) {
		for (int i = 0; i < N; i++)
			spins[i] = ((xorshift32(&randnum) & 1) << 1) - 1; 
		#pragma loop_coalesce 2
		for (int i = 0; i < N; i++) {
			int s = 0;
			for (int j = 0; j < N; j++) {
				int c = couplings[i*N+j]; // prefetching LSU
				int d = spins[j]; // prefetching LSU
				s += c*d; 
			}
			difference[i] = (-1) * s * spins[i];
		}

		// look up schedule
		beta = 0.1;
		for (int i = 0; i < SWEEP; i++) {
			float r = xorshift32(&randnum) / (float) MAX;
			schedule[i] = log(r) / beta;
			beta += increase;
		}

		// annealing
		for (int i = 0; i < SWEEP; i++) {
			int cur_schedule = schedule[i];
			for (int n = 0; n < N; n++) {
				int diff = difference[n];
				if (diff > cur_schedule) {
					spins[n] = -spins[n];
					difference[n] = -diff;
					int tmp = spins[n];
					#pragma unroll 8
					for (int j = 0; j < N; j++) {
						int c = couplings[n*N+j]; // prefetching LSU
						int d = spins[j]; // prefetching LSU
						difference[j] -= 2*c*d*tmp;
					}
				}
			}
		}

		// calculate result
		int E = 0;
		#pragma loop_coalesce 2
		for (int i = 0; i < N; i++) {
			int d = spins[i];
			for (int j = i; j < N; j++) {
				int c = couplings[i*N+j];
				E += d*c*spins[j];
			}
		}
		results[t] = -E;
	}
}

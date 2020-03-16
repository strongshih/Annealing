#define N 512
#define SWEEP 200
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

__kernel void ising(__global int* couplings, __global int* results)
{
	// random number
	int idx = get_global_id(0);
	uint randnum = idx + 1337;
	float beta = 0.1; // from 0.1 to 3.0
	float increase = (3.0 - 0.1) / (float) SWEEP;

	// spin initialization
	int spins[N];
	for (int i = 0; i < N; i++)
		spins[i] = ((xorshift32(&randnum) & 1) << 1) - 1; 
	

	// annealing
	for (int i = 0; i < SWEEP; i++) {
		beta += increase;
		float r = xorshift32(&randnum) / (float) MAX;
		for (int n = 0; n < N; n++) {
			int difference = 0;
			for (int j = 0; j < N; j++)
				difference += couplings[n*N+j]*spins[j];
			difference = -1 * difference * spins[n];
			if ((difference * beta) > log(r)) {
				spins[n] = -spins[n];
			}
		}
	}

	// calculate result
	int E = 0;
	for (int i = 0; i < N; i++)
		for (int j = i; j < N; j++)
			E += spins[i]*spins[j]*couplings[i*N+j];
	results[idx] = -E;
}

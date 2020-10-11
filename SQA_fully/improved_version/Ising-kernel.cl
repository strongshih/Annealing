#define bitshift 12
#define N (1 << bitshift)
#define bitmask N-1
#define M 4
#define P 8

__kernel void sqa(__global char *restrict couplings,
                  __global float *restrict randomLog,
                  __global float *restrict Jtrans,
                  __global char *restrict SPIN_IN, 
                  __global char *restrict SPIN_OUT)
{
    __private unsigned int LOOPCNT = (N+M-1)*(N/P);
    __private int lfield[M];
    __private char localJ[M][N+1];
    __private char lspin[M][N];
    __private float lJtrans;
    __private float lrandom[M][N];

    lJtrans = Jtrans[0];

    // copy global memory data to the internal memory;
    for (int j = 0; j < N; j++)
        # pragma unroll
        for (int i = 0; i < M; i++) {
            lspin[i][j] = SPIN_IN[(i<<bitshift)+j];
			lrandom[i][j] = randomLog[(i<<bitshift)+j];
		}
    
    // initialization
    #pragma unroll
    for (int i = 0; i < M; i++)
        lfield[i] = 0;

    #pragma ivdep
    for (int count = 0; count < LOOPCNT; count++) {
        // shift registers
        #pragma unroll
        for (int m = 0; m < M; m++) {
            #pragma unroll
            for (int n = N-1; n >= P; n--) {
                localJ[m][n] = localJ[m][n-P];
            }
        }

        // copying interaction coefficient data
		unsigned int current_idx = count*P;
		unsigned int index = ((current_idx >> bitshift) << bitshift) +
							  (current_idx & bitmask);
        #pragma unroll
        for (int p = 0; p < P; p++) {
            // Jij, (current_idx & bitmask) --> j,
            //      (current_idx >> bitshift) << bitshift) --> i
            localJ[0][p] = couplings[index+p];
        }

        // copying interaction coefficient to the next Trotter slice
        #pragma unroll
        for (int m = 0; m < M-1; m++)
            #pragma unroll
            for (int p = 0; p < P; p++)
                localJ[m+1][p] = localJ[m][((N/P)-1)*P+p]; // (N/P)-1) last chunk

        int klocal = ((count*P) >> bitshift); 

        #pragma unroll
        for (int m = 0; m < M; m++) {
            int idx = count*P;
			int j = idx & bitmask;
#if P == 8
            { 
                // parallel reduction
                int temp0 = localJ[m][0]*lspin[m][j+0]; 
                int temp1 = localJ[m][1]*lspin[m][j+1]; 
                int temp2 = localJ[m][2]*lspin[m][j+2]; 
                int temp3 = localJ[m][3]*lspin[m][j+3]; 
                int temp4 = localJ[m][4]*lspin[m][j+4]; 
                int temp5 = localJ[m][5]*lspin[m][j+5]; 
                int temp6 = localJ[m][6]*lspin[m][j+6]; 
                int temp7 = localJ[m][7]*lspin[m][j+7]; 

                int temp8 = temp0 + temp4;
                int temp9 = temp1 + temp5;
                int temp10 = temp2 + temp6;
                int temp11 = temp3 + temp7;

				int temp12 = temp8 + temp9;
				int temp13 = temp10 + temp11;

				int temp14 = temp12 + temp13;

                lfield[m] += temp14;
            }
#endif
#if P == 16
            { 
                // parallel reduction
                int temp0  = localJ[m][0]  * lspin[m][j+0]; 
                int temp1  = localJ[m][1]  * lspin[m][j+1]; 
                int temp2  = localJ[m][2]  * lspin[m][j+2]; 
                int temp3  = localJ[m][3]  * lspin[m][j+3]; 
                int temp4  = localJ[m][4]  * lspin[m][j+4]; 
                int temp5  = localJ[m][5]  * lspin[m][j+5]; 
                int temp6  = localJ[m][6]  * lspin[m][j+6]; 
                int temp7  = localJ[m][7]  * lspin[m][j+7]; 
                int temp8  = localJ[m][8]  * lspin[m][j+8]; 
                int temp9  = localJ[m][9]  * lspin[m][j+9]; 
                int temp10 = localJ[m][10] * lspin[m][j+10]; 
                int temp11 = localJ[m][11] * lspin[m][j+11]; 
                int temp12 = localJ[m][12] * lspin[m][j+12]; 
                int temp13 = localJ[m][13] * lspin[m][j+13]; 
                int temp14 = localJ[m][14] * lspin[m][j+14]; 
                int temp15 = localJ[m][15] * lspin[m][j+15]; 

                int temp16 = temp0 + temp8;
                int temp17 = temp1 + temp9;
                int temp18 = temp2 + temp10;
                int temp19 = temp3 + temp11;
                int temp20 = temp4 + temp12;
                int temp21 = temp5 + temp13;
                int temp22 = temp6 + temp14;
                int temp23 = temp7 + temp15;

				int temp24 = temp16 + temp20;
				int temp25 = temp17 + temp21;
				int temp26 = temp18 + temp22;
				int temp27 = temp19 + temp23;

				int temp28 = temp24 + temp26;
				int temp29 = temp25 + temp27;

				int temp30 = temp28 + temp29;

                lfield[m] += temp30;
            }
#endif
#if P == 32
            { 
                // parallel reduction
                int temp0  = localJ[m][0]  * lspin[m][j+0]; 
                int temp1  = localJ[m][1]  * lspin[m][j+1]; 
                int temp2  = localJ[m][2]  * lspin[m][j+2]; 
                int temp3  = localJ[m][3]  * lspin[m][j+3]; 
                int temp4  = localJ[m][4]  * lspin[m][j+4]; 
                int temp5  = localJ[m][5]  * lspin[m][j+5]; 
                int temp6  = localJ[m][6]  * lspin[m][j+6]; 
                int temp7  = localJ[m][7]  * lspin[m][j+7]; 
                int temp8  = localJ[m][8]  * lspin[m][j+8]; 
                int temp9  = localJ[m][9]  * lspin[m][j+9]; 
                int temp10 = localJ[m][10] * lspin[m][j+10]; 
                int temp11 = localJ[m][11] * lspin[m][j+11]; 
                int temp12 = localJ[m][12] * lspin[m][j+12]; 
                int temp13 = localJ[m][13] * lspin[m][j+13]; 
                int temp14 = localJ[m][14] * lspin[m][j+14]; 
                int temp15 = localJ[m][15] * lspin[m][j+15]; 
                int temp16 = localJ[m][16] * lspin[m][j+16]; 
                int temp17 = localJ[m][17] * lspin[m][j+17]; 
                int temp18 = localJ[m][18] * lspin[m][j+18]; 
                int temp19 = localJ[m][19] * lspin[m][j+19]; 
                int temp20 = localJ[m][20] * lspin[m][j+20]; 
                int temp21 = localJ[m][21] * lspin[m][j+21]; 
                int temp22 = localJ[m][22] * lspin[m][j+22]; 
                int temp23 = localJ[m][23] * lspin[m][j+23]; 
                int temp24 = localJ[m][24] * lspin[m][j+24]; 
                int temp25 = localJ[m][25] * lspin[m][j+25]; 
                int temp26 = localJ[m][26] * lspin[m][j+26]; 
                int temp27 = localJ[m][27] * lspin[m][j+27]; 
                int temp28 = localJ[m][28] * lspin[m][j+28]; 
                int temp29 = localJ[m][29] * lspin[m][j+29]; 
                int temp30 = localJ[m][30] * lspin[m][j+30]; 
                int temp31 = localJ[m][31] * lspin[m][j+31]; 

                int temp32 = temp0 + temp16;
                int temp33 = temp1 + temp17;
                int temp34 = temp2 + temp18;
                int temp35 = temp3 + temp19;
                int temp36 = temp4 + temp20;
                int temp37 = temp5 + temp21;
                int temp38 = temp6 + temp22;
                int temp39 = temp7 + temp23;
                int temp40 = temp8 + temp24;
                int temp41 = temp9 + temp25;
                int temp42 = temp10 + temp26;
                int temp43 = temp11 + temp27;
                int temp44 = temp12 + temp28;
                int temp45 = temp13 + temp29;
                int temp46 = temp14 + temp30;
                int temp47 = temp15 + temp31;

				int temp48 = temp32 + temp40;
				int temp49 = temp33 + temp41;
				int temp50 = temp34 + temp42;
				int temp51 = temp35 + temp43;
				int temp52 = temp36 + temp44;
				int temp53 = temp37 + temp45;
				int temp54 = temp38 + temp46;
				int temp55 = temp39 + temp47;

				int temp56 = temp48 + temp52;
				int temp57 = temp49 + temp53;
				int temp58 = temp50 + temp54;
				int temp59 = temp51 + temp55;

				int temp60 = temp56 + temp58;
				int temp61 = temp57 + temp59;

				int temp62 = temp60 + temp61;

                lfield[m] += temp62;
            }
#endif
#if P == 64
            {
                int temp0 = localJ[m][0] * lspin[m][j+0];
                int temp1 = localJ[m][1] * lspin[m][j+1];
                int temp2 = localJ[m][2] * lspin[m][j+2];
                int temp3 = localJ[m][3] * lspin[m][j+3];
                int temp4 = localJ[m][4] * lspin[m][j+4];
                int temp5 = localJ[m][5] * lspin[m][j+5];
                int temp6 = localJ[m][6] * lspin[m][j+6];
                int temp7 = localJ[m][7] * lspin[m][j+7];
                int temp8 = localJ[m][8] * lspin[m][j+8];
                int temp9 = localJ[m][9] * lspin[m][j+9];
                int temp10 = localJ[m][10] * lspin[m][j+10];
                int temp11 = localJ[m][11] * lspin[m][j+11];
                int temp12 = localJ[m][12] * lspin[m][j+12];
                int temp13 = localJ[m][13] * lspin[m][j+13];
                int temp14 = localJ[m][14] * lspin[m][j+14];
                int temp15 = localJ[m][15] * lspin[m][j+15];
                int temp16 = localJ[m][16] * lspin[m][j+16];
                int temp17 = localJ[m][17] * lspin[m][j+17];
                int temp18 = localJ[m][18] * lspin[m][j+18];
                int temp19 = localJ[m][19] * lspin[m][j+19];
                int temp20 = localJ[m][20] * lspin[m][j+20];
                int temp21 = localJ[m][21] * lspin[m][j+21];
                int temp22 = localJ[m][22] * lspin[m][j+22];
                int temp23 = localJ[m][23] * lspin[m][j+23];
                int temp24 = localJ[m][24] * lspin[m][j+24];
                int temp25 = localJ[m][25] * lspin[m][j+25];
                int temp26 = localJ[m][26] * lspin[m][j+26];
                int temp27 = localJ[m][27] * lspin[m][j+27];
                int temp28 = localJ[m][28] * lspin[m][j+28];
                int temp29 = localJ[m][29] * lspin[m][j+29];
                int temp30 = localJ[m][30] * lspin[m][j+30];
                int temp31 = localJ[m][31] * lspin[m][j+31];
                int temp32 = localJ[m][32] * lspin[m][j+32];
                int temp33 = localJ[m][33] * lspin[m][j+33];
                int temp34 = localJ[m][34] * lspin[m][j+34];
                int temp35 = localJ[m][35] * lspin[m][j+35];
                int temp36 = localJ[m][36] * lspin[m][j+36];
                int temp37 = localJ[m][37] * lspin[m][j+37];
                int temp38 = localJ[m][38] * lspin[m][j+38];
                int temp39 = localJ[m][39] * lspin[m][j+39];
                int temp40 = localJ[m][40] * lspin[m][j+40];
                int temp41 = localJ[m][41] * lspin[m][j+41];
                int temp42 = localJ[m][42] * lspin[m][j+42];
                int temp43 = localJ[m][43] * lspin[m][j+43];
                int temp44 = localJ[m][44] * lspin[m][j+44];
                int temp45 = localJ[m][45] * lspin[m][j+45];
                int temp46 = localJ[m][46] * lspin[m][j+46];
                int temp47 = localJ[m][47] * lspin[m][j+47];
                int temp48 = localJ[m][48] * lspin[m][j+48];
                int temp49 = localJ[m][49] * lspin[m][j+49];
                int temp50 = localJ[m][50] * lspin[m][j+50];
                int temp51 = localJ[m][51] * lspin[m][j+51];
                int temp52 = localJ[m][52] * lspin[m][j+52];
                int temp53 = localJ[m][53] * lspin[m][j+53];
                int temp54 = localJ[m][54] * lspin[m][j+54];
                int temp55 = localJ[m][55] * lspin[m][j+55];
                int temp56 = localJ[m][56] * lspin[m][j+56];
                int temp57 = localJ[m][57] * lspin[m][j+57];
                int temp58 = localJ[m][58] * lspin[m][j+58];
                int temp59 = localJ[m][59] * lspin[m][j+59];
                int temp60 = localJ[m][60] * lspin[m][j+60];
                int temp61 = localJ[m][61] * lspin[m][j+61];
                int temp62 = localJ[m][62] * lspin[m][j+62];
                int temp63 = localJ[m][63] * lspin[m][j+63];

                int temp64 = temp0 + temp32;
                int temp65 = temp1 + temp33;
                int temp66 = temp2 + temp34;
                int temp67 = temp3 + temp35;
                int temp68 = temp4 + temp36;
                int temp69 = temp5 + temp37;
                int temp70 = temp6 + temp38;
                int temp71 = temp7 + temp39;
                int temp72 = temp8 + temp40;
                int temp73 = temp9 + temp41;
                int temp74 = temp10 + temp42;
                int temp75 = temp11 + temp43;
                int temp76 = temp12 + temp44;
                int temp77 = temp13 + temp45;
                int temp78 = temp14 + temp46;
                int temp79 = temp15 + temp47;
                int temp80 = temp16 + temp48;
                int temp81 = temp17 + temp49;
                int temp82 = temp18 + temp50;
                int temp83 = temp19 + temp51;
                int temp84 = temp20 + temp52;
                int temp85 = temp21 + temp53;
                int temp86 = temp22 + temp54;
                int temp87 = temp23 + temp55;
                int temp88 = temp24 + temp56;
                int temp89 = temp25 + temp57;
                int temp90 = temp26 + temp58;
                int temp91 = temp27 + temp59;
                int temp92 = temp28 + temp60;
                int temp93 = temp29 + temp61;
                int temp94 = temp30 + temp62;
                int temp95 = temp31 + temp63;

                int temp96 = temp64 + temp80;
                int temp97 = temp65 + temp81;
                int temp98 = temp66 + temp82;
                int temp99 = temp67 + temp83;
                int temp100 = temp68 + temp84;
                int temp101 = temp69 + temp85;
                int temp102 = temp70 + temp86;
                int temp103 = temp71 + temp87;
                int temp104 = temp72 + temp88;
                int temp105 = temp73 + temp89;
                int temp106 = temp74 + temp90;
                int temp107 = temp75 + temp91;
                int temp108 = temp76 + temp92;
                int temp109 = temp77 + temp93;
                int temp110 = temp78 + temp94;
                int temp111 = temp79 + temp95;

                int temp112 = temp96 + temp104;
                int temp113 = temp97 + temp105;
                int temp114 = temp98 + temp106;
                int temp115 = temp99 + temp107;
                int temp116 = temp100 + temp108;
                int temp117 = temp101 + temp109;
                int temp118 = temp102 + temp110;
                int temp119 = temp103 + temp111;

                int temp120 = temp112 + temp116;
                int temp121 = temp113 + temp117;
                int temp122 = temp114 + temp118;
                int temp123 = temp115 + temp119;

                int temp124 = temp120 + temp122;
                int temp125 = temp121 + temp123;

                int temp126 = temp124 + temp125;

                lfield[m] += temp126;
			}
#endif
#if P == 128
            {
                int temp0 = localJ[m][0] * lspin[m][j+0];
                int temp1 = localJ[m][1] * lspin[m][j+1];
                int temp2 = localJ[m][2] * lspin[m][j+2];
                int temp3 = localJ[m][3] * lspin[m][j+3];
                int temp4 = localJ[m][4] * lspin[m][j+4];
                int temp5 = localJ[m][5] * lspin[m][j+5];
                int temp6 = localJ[m][6] * lspin[m][j+6];
                int temp7 = localJ[m][7] * lspin[m][j+7];
                int temp8 = localJ[m][8] * lspin[m][j+8];
                int temp9 = localJ[m][9] * lspin[m][j+9];
                int temp10 = localJ[m][10] * lspin[m][j+10];
                int temp11 = localJ[m][11] * lspin[m][j+11];
                int temp12 = localJ[m][12] * lspin[m][j+12];
                int temp13 = localJ[m][13] * lspin[m][j+13];
                int temp14 = localJ[m][14] * lspin[m][j+14];
                int temp15 = localJ[m][15] * lspin[m][j+15];
                int temp16 = localJ[m][16] * lspin[m][j+16];
                int temp17 = localJ[m][17] * lspin[m][j+17];
                int temp18 = localJ[m][18] * lspin[m][j+18];
                int temp19 = localJ[m][19] * lspin[m][j+19];
                int temp20 = localJ[m][20] * lspin[m][j+20];
                int temp21 = localJ[m][21] * lspin[m][j+21];
                int temp22 = localJ[m][22] * lspin[m][j+22];
                int temp23 = localJ[m][23] * lspin[m][j+23];
                int temp24 = localJ[m][24] * lspin[m][j+24];
                int temp25 = localJ[m][25] * lspin[m][j+25];
                int temp26 = localJ[m][26] * lspin[m][j+26];
                int temp27 = localJ[m][27] * lspin[m][j+27];
                int temp28 = localJ[m][28] * lspin[m][j+28];
                int temp29 = localJ[m][29] * lspin[m][j+29];
                int temp30 = localJ[m][30] * lspin[m][j+30];
                int temp31 = localJ[m][31] * lspin[m][j+31];
                int temp32 = localJ[m][32] * lspin[m][j+32];
                int temp33 = localJ[m][33] * lspin[m][j+33];
                int temp34 = localJ[m][34] * lspin[m][j+34];
                int temp35 = localJ[m][35] * lspin[m][j+35];
                int temp36 = localJ[m][36] * lspin[m][j+36];
                int temp37 = localJ[m][37] * lspin[m][j+37];
                int temp38 = localJ[m][38] * lspin[m][j+38];
                int temp39 = localJ[m][39] * lspin[m][j+39];
                int temp40 = localJ[m][40] * lspin[m][j+40];
                int temp41 = localJ[m][41] * lspin[m][j+41];
                int temp42 = localJ[m][42] * lspin[m][j+42];
                int temp43 = localJ[m][43] * lspin[m][j+43];
                int temp44 = localJ[m][44] * lspin[m][j+44];
                int temp45 = localJ[m][45] * lspin[m][j+45];
                int temp46 = localJ[m][46] * lspin[m][j+46];
                int temp47 = localJ[m][47] * lspin[m][j+47];
                int temp48 = localJ[m][48] * lspin[m][j+48];
                int temp49 = localJ[m][49] * lspin[m][j+49];
                int temp50 = localJ[m][50] * lspin[m][j+50];
                int temp51 = localJ[m][51] * lspin[m][j+51];
                int temp52 = localJ[m][52] * lspin[m][j+52];
                int temp53 = localJ[m][53] * lspin[m][j+53];
                int temp54 = localJ[m][54] * lspin[m][j+54];
                int temp55 = localJ[m][55] * lspin[m][j+55];
                int temp56 = localJ[m][56] * lspin[m][j+56];
                int temp57 = localJ[m][57] * lspin[m][j+57];
                int temp58 = localJ[m][58] * lspin[m][j+58];
                int temp59 = localJ[m][59] * lspin[m][j+59];
                int temp60 = localJ[m][60] * lspin[m][j+60];
                int temp61 = localJ[m][61] * lspin[m][j+61];
                int temp62 = localJ[m][62] * lspin[m][j+62];
                int temp63 = localJ[m][63] * lspin[m][j+63];
                int temp64 = localJ[m][64] * lspin[m][j+64];
                int temp65 = localJ[m][65] * lspin[m][j+65];
                int temp66 = localJ[m][66] * lspin[m][j+66];
                int temp67 = localJ[m][67] * lspin[m][j+67];
                int temp68 = localJ[m][68] * lspin[m][j+68];
                int temp69 = localJ[m][69] * lspin[m][j+69];
                int temp70 = localJ[m][70] * lspin[m][j+70];
                int temp71 = localJ[m][71] * lspin[m][j+71];
                int temp72 = localJ[m][72] * lspin[m][j+72];
                int temp73 = localJ[m][73] * lspin[m][j+73];
                int temp74 = localJ[m][74] * lspin[m][j+74];
                int temp75 = localJ[m][75] * lspin[m][j+75];
                int temp76 = localJ[m][76] * lspin[m][j+76];
                int temp77 = localJ[m][77] * lspin[m][j+77];
                int temp78 = localJ[m][78] * lspin[m][j+78];
                int temp79 = localJ[m][79] * lspin[m][j+79];
                int temp80 = localJ[m][80] * lspin[m][j+80];
                int temp81 = localJ[m][81] * lspin[m][j+81];
                int temp82 = localJ[m][82] * lspin[m][j+82];
                int temp83 = localJ[m][83] * lspin[m][j+83];
                int temp84 = localJ[m][84] * lspin[m][j+84];
                int temp85 = localJ[m][85] * lspin[m][j+85];
                int temp86 = localJ[m][86] * lspin[m][j+86];
                int temp87 = localJ[m][87] * lspin[m][j+87];
                int temp88 = localJ[m][88] * lspin[m][j+88];
                int temp89 = localJ[m][89] * lspin[m][j+89];
                int temp90 = localJ[m][90] * lspin[m][j+90];
                int temp91 = localJ[m][91] * lspin[m][j+91];
                int temp92 = localJ[m][92] * lspin[m][j+92];
                int temp93 = localJ[m][93] * lspin[m][j+93];
                int temp94 = localJ[m][94] * lspin[m][j+94];
                int temp95 = localJ[m][95] * lspin[m][j+95];
                int temp96 = localJ[m][96] * lspin[m][j+96];
                int temp97 = localJ[m][97] * lspin[m][j+97];
                int temp98 = localJ[m][98] * lspin[m][j+98];
                int temp99 = localJ[m][99] * lspin[m][j+99];
                int temp100 = localJ[m][100] * lspin[m][j+100];
                int temp101 = localJ[m][101] * lspin[m][j+101];
                int temp102 = localJ[m][102] * lspin[m][j+102];
                int temp103 = localJ[m][103] * lspin[m][j+103];
                int temp104 = localJ[m][104] * lspin[m][j+104];
                int temp105 = localJ[m][105] * lspin[m][j+105];
                int temp106 = localJ[m][106] * lspin[m][j+106];
                int temp107 = localJ[m][107] * lspin[m][j+107];
                int temp108 = localJ[m][108] * lspin[m][j+108];
                int temp109 = localJ[m][109] * lspin[m][j+109];
                int temp110 = localJ[m][110] * lspin[m][j+110];
                int temp111 = localJ[m][111] * lspin[m][j+111];
                int temp112 = localJ[m][112] * lspin[m][j+112];
                int temp113 = localJ[m][113] * lspin[m][j+113];
                int temp114 = localJ[m][114] * lspin[m][j+114];
                int temp115 = localJ[m][115] * lspin[m][j+115];
                int temp116 = localJ[m][116] * lspin[m][j+116];
                int temp117 = localJ[m][117] * lspin[m][j+117];
                int temp118 = localJ[m][118] * lspin[m][j+118];
                int temp119 = localJ[m][119] * lspin[m][j+119];
                int temp120 = localJ[m][120] * lspin[m][j+120];
                int temp121 = localJ[m][121] * lspin[m][j+121];
                int temp122 = localJ[m][122] * lspin[m][j+122];
                int temp123 = localJ[m][123] * lspin[m][j+123];
                int temp124 = localJ[m][124] * lspin[m][j+124];
                int temp125 = localJ[m][125] * lspin[m][j+125];
                int temp126 = localJ[m][126] * lspin[m][j+126];
                int temp127 = localJ[m][127] * lspin[m][j+127];

                int temp128 = temp0 + temp64;
                int temp129 = temp1 + temp65;
                int temp130 = temp2 + temp66;
                int temp131 = temp3 + temp67;
                int temp132 = temp4 + temp68;
                int temp133 = temp5 + temp69;
                int temp134 = temp6 + temp70;
                int temp135 = temp7 + temp71;
                int temp136 = temp8 + temp72;
                int temp137 = temp9 + temp73;
                int temp138 = temp10 + temp74;
                int temp139 = temp11 + temp75;
                int temp140 = temp12 + temp76;
                int temp141 = temp13 + temp77;
                int temp142 = temp14 + temp78;
                int temp143 = temp15 + temp79;
                int temp144 = temp16 + temp80;
                int temp145 = temp17 + temp81;
                int temp146 = temp18 + temp82;
                int temp147 = temp19 + temp83;
                int temp148 = temp20 + temp84;
                int temp149 = temp21 + temp85;
                int temp150 = temp22 + temp86;
                int temp151 = temp23 + temp87;
                int temp152 = temp24 + temp88;
                int temp153 = temp25 + temp89;
                int temp154 = temp26 + temp90;
                int temp155 = temp27 + temp91;
                int temp156 = temp28 + temp92;
                int temp157 = temp29 + temp93;
                int temp158 = temp30 + temp94;
                int temp159 = temp31 + temp95;
                int temp160 = temp32 + temp96;
                int temp161 = temp33 + temp97;
                int temp162 = temp34 + temp98;
                int temp163 = temp35 + temp99;
                int temp164 = temp36 + temp100;
                int temp165 = temp37 + temp101;
                int temp166 = temp38 + temp102;
                int temp167 = temp39 + temp103;
                int temp168 = temp40 + temp104;
                int temp169 = temp41 + temp105;
                int temp170 = temp42 + temp106;
                int temp171 = temp43 + temp107;
                int temp172 = temp44 + temp108;
                int temp173 = temp45 + temp109;
                int temp174 = temp46 + temp110;
                int temp175 = temp47 + temp111;
                int temp176 = temp48 + temp112;
                int temp177 = temp49 + temp113;
                int temp178 = temp50 + temp114;
                int temp179 = temp51 + temp115;
                int temp180 = temp52 + temp116;
                int temp181 = temp53 + temp117;
                int temp182 = temp54 + temp118;
                int temp183 = temp55 + temp119;
                int temp184 = temp56 + temp120;
                int temp185 = temp57 + temp121;
                int temp186 = temp58 + temp122;
                int temp187 = temp59 + temp123;
                int temp188 = temp60 + temp124;
                int temp189 = temp61 + temp125;
                int temp190 = temp62 + temp126;
                int temp191 = temp63 + temp127;

                int temp192 = temp128 + temp160;
                int temp193 = temp129 + temp161;
                int temp194 = temp130 + temp162;
                int temp195 = temp131 + temp163;
                int temp196 = temp132 + temp164;
                int temp197 = temp133 + temp165;
                int temp198 = temp134 + temp166;
                int temp199 = temp135 + temp167;
                int temp200 = temp136 + temp168;
                int temp201 = temp137 + temp169;
                int temp202 = temp138 + temp170;
                int temp203 = temp139 + temp171;
                int temp204 = temp140 + temp172;
                int temp205 = temp141 + temp173;
                int temp206 = temp142 + temp174;
                int temp207 = temp143 + temp175;
                int temp208 = temp144 + temp176;
                int temp209 = temp145 + temp177;
                int temp210 = temp146 + temp178;
                int temp211 = temp147 + temp179;
                int temp212 = temp148 + temp180;
                int temp213 = temp149 + temp181;
                int temp214 = temp150 + temp182;
                int temp215 = temp151 + temp183;
                int temp216 = temp152 + temp184;
                int temp217 = temp153 + temp185;
                int temp218 = temp154 + temp186;
                int temp219 = temp155 + temp187;
                int temp220 = temp156 + temp188;
                int temp221 = temp157 + temp189;
                int temp222 = temp158 + temp190;
                int temp223 = temp159 + temp191;

                int temp224 = temp192 + temp208;
                int temp225 = temp193 + temp209;
                int temp226 = temp194 + temp210;
                int temp227 = temp195 + temp211;
                int temp228 = temp196 + temp212;
                int temp229 = temp197 + temp213;
                int temp230 = temp198 + temp214;
                int temp231 = temp199 + temp215;
                int temp232 = temp200 + temp216;
                int temp233 = temp201 + temp217;
                int temp234 = temp202 + temp218;
                int temp235 = temp203 + temp219;
                int temp236 = temp204 + temp220;
                int temp237 = temp205 + temp221;
                int temp238 = temp206 + temp222;
                int temp239 = temp207 + temp223;

                int temp240 = temp224 + temp232;
                int temp241 = temp225 + temp233;
                int temp242 = temp226 + temp234;
                int temp243 = temp227 + temp235;
                int temp244 = temp228 + temp236;
                int temp245 = temp229 + temp237;
                int temp246 = temp230 + temp238;
                int temp247 = temp231 + temp239;

                int temp248 = temp240 + temp244;
                int temp249 = temp241 + temp245;
                int temp250 = temp242 + temp246;
                int temp251 = temp243 + temp247;

                int temp252 = temp248 + temp250;
                int temp253 = temp249 + temp251;

                int temp254 = temp252 + temp253;

                lfield[m] += temp254;
			}
#endif

			int kklocal = klocal - m; 
			unsigned int up = (m!=0) ? m-1 : M-1;
			unsigned int down = (m!=M-1) ? m+1 : 0;
			char u = lspin[up][kklocal];
			char d = lspin[down][kklocal];
			char cur = lspin[m][kklocal];
			float tmp1 = lfield[m];
			float tmp2 = tmp1-lJtrans*(u + d);
			float diff = cur * tmp2; 

            if ( ((count+1)*P & bitmask) == 0 ) {
                if ( diff < lrandom[m][kklocal] )
                    lspin[m][kklocal] = -cur;
                lfield[m] = 0;
            }
        }
    }

    // copy spin values from the internal memory to the global memory
    for (int j = 0; j < N; j++)
        # pragma unroll
        for (int i = 0; i < M; i++)
            SPIN_OUT[(i<<bitshift)+j] = lspin[i][j];
}

// #include "../common/npb-CPP.hpp"
#include "../../libs/headers/npb-CPP.hpp"
#include "../npbparams.hpp"
#include <cstring>

#define MK 16
#define MM (M - MK)
#define NN (1 << MM)
#define NK (1 << MK)
#define NQ 10
#define EPSILON 1.0e-8
#define A 1220703125.0
#define S 271828183.0
#define NK_PLUS ((2*NK)+1)

#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static double x[NK_PLUS];
static double q[NQ];
#else
static double (*x)=(double*)malloc(sizeof(double)*(NK_PLUS));
static double (*q)=(double*)malloc(sizeof(double)*(NQ));
#endif

static void dump_result(const char* filename, double sx, double sy, double* q);

int main(int argc, char **argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
	double Mops, t1, t2, t3, t4, x1, x2;
	double sx, sy, tm, an, tt, gc;
	int    np;
	int    i, ik, kk, l, k, nit;
	int    k_offset, j;
	boolean timers_enabled;
	double dum[3] = {1.0, 1.0, 1.0};
	char   size[16];

	// ── 解析 verbose 参数 ─────────────────────────────────
	boolean verbose = FALSE;
	for(int a = 1; a < argc; a++){
		if(strcmp(argv[a], "verbose") == 0){
			verbose = TRUE;
			break;
		}
	}
	// ─────────────────────────────────────────────────────

	FILE* fp;
	if((fp = fopen("timer.flag", "r")) == NULL){
		timers_enabled = FALSE;
	}else{
		timers_enabled = TRUE;
		fclose(fp);
	}

	np = NN;

	vranlc(0, &dum[0], dum[1], &dum[2]);
	dum[0] = randlc(&dum[1], dum[2]);
	for(i=0; i<NK_PLUS; i++){x[i] = -1.0e99;}
	Mops = log(sqrt(fabs(max(1.0, 1.0))));

	timer_clear(0);
	timer_clear(1);
	timer_clear(2);
	timer_start(0);

	t1 = A;
	vranlc(0, &t1, A, x);

	t1 = A;
	for(i=0; i<MK+1; i++){
		t2 = randlc(&t1, t1);
	}

	an = t1;
	tt = S;
	gc = 0.0;
	sx = 0.0;
	sy = 0.0;

	for(i=0; i<=NQ-1; i++){
		q[i] = 0.0;
	}

	k_offset = -1;

	for(k=1; k<=np; k++){
		kk = k_offset + k;
		t1 = S;
		t2 = an;

		for(i=1; i<=100; i++){
			ik = kk / 2;
			if((2*ik) != kk){t3 = randlc(&t1, t2);}
			if(ik == 0){break;}
			t3 = randlc(&t2, t2);
			kk = ik;
		}

		if(timers_enabled){timer_start(2);}
		vranlc(2*NK, &t1, A, x);
		if(timers_enabled){timer_stop(2);}

		if(timers_enabled){timer_start(1);}
		for(i=0; i<NK; i++){
			x1 = 2.0 * x[2*i] - 1.0;
			x2 = 2.0 * x[2*i+1] - 1.0;
			t1 = pow2(x1) + pow2(x2);
			if(t1 <= 1.0){
				t2 = sqrt(-2.0 * log(t1) / t1);
				t3 = (x1 * t2);
				t4 = (x2 * t2);
				l = max(fabs(t3), fabs(t4));
				q[l] += 1.0;
				sx = sx + t3;
				sy = sy + t4;
			}
		}
		if(timers_enabled){timer_stop(1);}
	}

	for(i=0; i<=NQ-1; i++){
		gc = gc + q[i];
	}

	timer_stop(0);
	tm = timer_read(0);

	// ── 始终只输出总耗时 ──────────────────────────────────
	printf("Time: %.6f(s)\n", tm);
	// ─────────────────────────────────────────────────────

	Mops = pow(2.0, M+1) / tm / 1000000.0;

	if(tm != 0.0){
		nit = 0;
	}

	// ── verbose 模式：dump 结果 + 写 ep_out.txt ───────────
	if(verbose){
		dump_result("ep_out.txt", sx, sy, q);

		FILE* fout = fopen("ep_ref.txt", "w");
		if(fout == NULL){
			fprintf(stderr, "Error: cannot open ep_out.txt for writing\n");
		}else{
			fprintf(fout, "class      = %c\n", CLASS);
			fprintf(fout, "m          = %d\n", M);
			fprintf(fout, "time(s)    = %.6f\n", tm);
			fprintf(fout, "mops       = %.2f\n", Mops);
			fclose(fout);
		}
	}
	// ─────────────────────────────────────────────────────

	return 0;
}

void dump_result(const char* filename, double sx, double sy, double* q){
	FILE* fp = fopen(filename, "w");
	if(fp == NULL){
		fprintf(stderr, "Error: cannot open %s for writing\n", filename);
		return;
	}
	// sx, sy
	fprintf(fp, "%.17f\n", sx);
	fprintf(fp, "%.17f\n", sy);
	// q 数组 NQ 个计数值
	for(int i=0; i<NQ; i++){
		fprintf(fp, "%.17f\n", q[i]);
	}
	fclose(fp);
}
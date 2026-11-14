// #include "../common/npb-CPP.hpp"
#include "../../libs/headers/npb-CPP.hpp"
#include "../npbparams.hpp"
#include <cstring>

#define FFTBLOCK_DEFAULT DEFAULT_BEHAVIOR
#define FFTBLOCKPAD_DEFAULT DEFAULT_BEHAVIOR
#define FFTBLOCK FFTBLOCK_DEFAULT
#define FFTBLOCKPAD FFTBLOCKPAD_DEFAULT
#define SEED 314159265.0
#define A 1220703125.0
#define PI 3.141592653589793238
#define ALPHA 1.0e-6
#define T_TOTAL 1
#define T_SETUP 2
#define T_FFT 3
#define T_EVOLVE 4
#define T_CHECKSUM 5
#define T_FFTX 6
#define T_FFTY 7
#define T_FFTZ 8
#define T_MAX 8

#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static dcomplex sums[NITER_DEFAULT+1];
static double twiddle[NTOTAL];
static dcomplex u[MAXDIM];
static dcomplex u0[NTOTAL];
static dcomplex u1[NTOTAL];
static int dims[3];
#else
static dcomplex (*sums)=(dcomplex*)malloc(sizeof(dcomplex)*(NITER_DEFAULT+1));
static double (*twiddle)=(double*)malloc(sizeof(double)*(NTOTAL));
static dcomplex (*u)=(dcomplex*)malloc(sizeof(dcomplex)*(MAXDIM));
static dcomplex (*u0)=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static dcomplex (*u1)=(dcomplex*)malloc(sizeof(dcomplex)*(NTOTAL));
static int (*dims)=(int*)malloc(sizeof(int)*(3));
#endif
static int niter;
static boolean timers_enabled;
static boolean debug;

static void cffts1(int is, int d1, int d2, int d3,
		void* pointer_x, void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD], dcomplex y2[][FFTBLOCKPAD]);
static void cffts2(int is, int d1, int d2, int d3,
		void* pointer_x, void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD], dcomplex y2[][FFTBLOCKPAD]);
static void cffts3(int is, int d1, int d2, int d3,
		void* pointer_x, void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD], dcomplex y2[][FFTBLOCKPAD]);
static void cfftz(int is, int m, int n,
		dcomplex x[][FFTBLOCKPAD], dcomplex y[][FFTBLOCKPAD]);
static void checksum(int i, void* pointer_u1, int d1, int d2, int d3);
static void compute_indexmap(void* pointer_twiddle, int d1, int d2, int d3);
static void compute_initial_conditions(void* pointer_u0, int d1, int d2, int d3);
static void evolve(void* pointer_u0, void* pointer_u1,
		void* pointer_twiddle, int d1, int d2, int d3);
static void fft(int dir, void* pointer_x1, void* pointer_x2);
static void fft_init(int n);
static void fftz2(int is, int l, int m, int n, int ny, int ny1,
		dcomplex u[], dcomplex x[][FFTBLOCKPAD], dcomplex y[][FFTBLOCKPAD]);
static int ilog2(int n);
static void init_ui(void* pointer_u0, void* pointer_u1,
		void* pointer_twiddle, int d1, int d2, int d3);
static void ipow46(double a, int exponent, double* result);
static void setup();
static void dump_result(const char* filename);

int main(int argc, char **argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
	int i, iter;
	double total_time, mflops;
	char class_npb;

	// ── 解析 verbose 参数 ─────────────────────────────────
	boolean verbose = FALSE;
	for(int a = 1; a < argc; a++){
		if(strcmp(argv[a], "verbose") == 0){
			verbose = TRUE;
			break;
		}
	}
	// ─────────────────────────────────────────────────────

	for(i=0; i<T_MAX; i++){timer_clear(i);}
	setup();
	init_ui(u0, u1, twiddle, dims[0], dims[1], dims[2]);
	compute_indexmap(twiddle, dims[0], dims[1], dims[2]);
	compute_initial_conditions(u1, dims[0], dims[1], dims[2]);
	fft_init(MAXDIM);
	fft(1, u1, u0);

	for(i=0; i<T_MAX; i++){timer_clear(i);}
	timer_start(T_TOTAL);
	if(timers_enabled==TRUE){timer_start(T_SETUP);}

	compute_indexmap(twiddle, dims[0], dims[1], dims[2]);
	compute_initial_conditions(u1, dims[0], dims[1], dims[2]);
	fft_init(MAXDIM);

	if(timers_enabled==TRUE){timer_stop(T_SETUP);}
	if(timers_enabled==TRUE){timer_start(T_FFT);}
	fft(1, u1, u0);
	if(timers_enabled==TRUE){timer_stop(T_FFT);}

	for(iter=1; iter<=niter; iter++){
		if(timers_enabled==TRUE){timer_start(T_EVOLVE);}
		evolve(u0, u1, twiddle, dims[0], dims[1], dims[2]);
		if(timers_enabled==TRUE){timer_stop(T_EVOLVE);}
		if(timers_enabled==TRUE){timer_start(T_FFT);}
		fft(-1, u1, u1);
		if(timers_enabled==TRUE){timer_stop(T_FFT);}
		if(timers_enabled==TRUE){timer_start(T_CHECKSUM);}
		checksum(iter, u1, dims[0], dims[1], dims[2]);
		if(timers_enabled==TRUE){timer_stop(T_CHECKSUM);}
	}

	timer_stop(T_TOTAL);
	total_time = timer_read(T_TOTAL);

	// ── 确定 class ────────────────────────────────────────
	if     (NX==64  && NY==64  && NZ==64  && niter==6)  class_npb='S';
	else if(NX==128 && NY==128 && NZ==32  && niter==6)  class_npb='W';
	else if(NX==256 && NY==256 && NZ==128 && niter==6)  class_npb='A';
	else if(NX==512 && NY==256 && NZ==256 && niter==20) class_npb='B';
	else if(NX==512 && NY==512 && NZ==512 && niter==20) class_npb='C';
	else if(NX==2048&& NY==1024&& NZ==1024&& niter==25) class_npb='D';
	else if(NX==4096&& NY==2048&& NZ==2048&& niter==25) class_npb='E';
	else                                                  class_npb='U';

	// ── 始终只输出总耗时 ──────────────────────────────────
	printf("Time: %.6f(s)\n", total_time);
	// ─────────────────────────────────────────────────────

	if(total_time != 0.0){
		mflops = 1.0e-6 * ((double)(NTOTAL)) *
			(14.8157 + 7.19641 * log((double)(NTOTAL))
			 + (5.23518 + 7.21113 * log((double)(NTOTAL)))*niter)
			/ total_time;
	}else{
		mflops = 0.0;
	}

	// ── verbose 模式：dump 结果 + 写 ft_out.txt ───────────
	if(verbose){
		dump_result("ft_out.txt");

		FILE* fout = fopen("ft_ref.txt", "w");
		if(fout == NULL){
			fprintf(stderr, "Error: cannot open ft_out.txt for writing\n");
		}else{
			fprintf(fout, "class      = %c\n", class_npb);
			fprintf(fout, "size       = %dx%dx%d\n", NX, NY, NZ);
			fprintf(fout, "iterations = %d\n", niter);
			fprintf(fout, "time(s)    = %.6f\n", total_time);
			fprintf(fout, "mflops     = %.2f\n", mflops);
			fclose(fout);
		}
	}
	// ─────────────────────────────────────────────────────

	return 0;
}

void dump_result(const char* filename){
	FILE* fp = fopen(filename, "w");
	if(fp == NULL){
		fprintf(stderr, "Error: cannot open %s for writing\n", filename);
		return;
	}
	// 每次迭代的 checksum，real 和 imag 各一行
	for(int i=1; i<=niter; i++){
		fprintf(fp, "%.17f\n", sums[i].real);
		fprintf(fp, "%.17f\n", sums[i].imag);
	}
	fclose(fp);
}

// ── 以下函数与原版完全相同，保持不变 ─────────────────────────

static void cffts1(int is, int d1, int d2, int d3,
		void* pointer_x, void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD], dcomplex y2[][FFTBLOCKPAD]){
	dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;
	int logd1, i, j, k, jj;
	logd1 = ilog2(d1);
	if(timers_enabled){timer_start(T_FFTX);}
	for(k=0; k<d3; k++){
		for(jj=0; jj<=d2-FFTBLOCK; jj+=FFTBLOCK){
			for(j=0; j<FFTBLOCK; j++)
				for(i=0; i<d1; i++)
					y1[i][j] = x[k][j+jj][i];
			cfftz(is, logd1, d1, y1, y2);
			for(j=0; j<FFTBLOCK; j++)
				for(i=0; i<d1; i++)
					xout[k][j+jj][i] = y1[i][j];
		}
	}
	if(timers_enabled){timer_stop(T_FFTX);}
}

static void cffts2(int is, int d1, int d2, int d3,
		void* pointer_x, void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD], dcomplex y2[][FFTBLOCKPAD]){
	dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;
	int logd2, i, j, k, ii;
	logd2 = ilog2(d2);
	if(timers_enabled){timer_start(T_FFTY);}
	for(k=0; k<d3; k++){
		for(ii=0; ii<=d1-FFTBLOCK; ii+=FFTBLOCK){
			for(j=0; j<d2; j++)
				for(i=0; i<FFTBLOCK; i++)
					y1[j][i] = x[k][j][i+ii];
			cfftz(is, logd2, d2, y1, y2);
			for(j=0; j<d2; j++)
				for(i=0; i<FFTBLOCK; i++)
					xout[k][j][i+ii] = y1[j][i];
		}
	}
	if(timers_enabled){timer_stop(T_FFTY);}
}

static void cffts3(int is, int d1, int d2, int d3,
		void* pointer_x, void* pointer_xout,
		dcomplex y1[][FFTBLOCKPAD], dcomplex y2[][FFTBLOCKPAD]){
	dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;
	int logd3, i, j, k, ii;
	logd3 = ilog2(d3);
	if(timers_enabled){timer_start(T_FFTZ);}
	for(j=0; j<d2; j++){
		for(ii=0; ii<=d1-FFTBLOCK; ii+=FFTBLOCK){
			for(k=0; k<d3; k++)
				for(i=0; i<FFTBLOCK; i++)
					y1[k][i] = x[k][j][i+ii];
			cfftz(is, logd3, d3, y1, y2);
			for(k=0; k<d3; k++)
				for(i=0; i<FFTBLOCK; i++)
					xout[k][j][i+ii] = y1[k][i];
		}
	}
	if(timers_enabled){timer_stop(T_FFTZ);}
}

static void cfftz(int is, int m, int n,
		dcomplex x[][FFTBLOCKPAD], dcomplex y[][FFTBLOCKPAD]){
	int i, j, l, mx;
	mx = (int)(u[0].real);
	if((is != 1 && is != -1) || m < 1 || m > mx){
		printf("CFFTZ: Either U has not been initialized, or else\n"
				"one of the input parameters is invalid%5d%5d%5d\n", is, m, mx);
		exit(EXIT_FAILURE);
	}
	for(l=1; l<=m; l+=2){
		fftz2(is, l, m, n, FFTBLOCK, FFTBLOCKPAD, u, x, y);
		if(l==m){
			for(j=0; j<n; j++)
				for(i=0; i<FFTBLOCK; i++)
					x[j][i] = y[j][i];
			break;
		}
		fftz2(is, l+1, m, n, FFTBLOCK, FFTBLOCKPAD, u, y, x);
	}
}

static void checksum(int i, void* pointer_u1, int d1, int d2, int d3){
	dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	int j, q, r, s;
	dcomplex chk = dcomplex_create(0.0, 0.0);
	for(j=1; j<=1024; j++){
		q = j % NX;
		r = (3*j) % NY;
		s = (5*j) % NZ;
		chk = dcomplex_add(chk, u1[s][r][q]);
	}
	chk = dcomplex_div2(chk, (double)(NTOTAL));
	sums[i] = chk;
}

static void compute_indexmap(void* pointer_twiddle, int d1, int d2, int d3){
	double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;
	int i, j, k, kk, kk2, jj, kj2, ii;
	double ap;
	ap = -4.0 * ALPHA * PI * PI;
	for(k=0; k<d3; k++){
		kk = ((k+NZ/2) % NZ) - NZ/2;
		kk2 = kk*kk;
		for(j=0; j<d2; j++){
			jj = ((j+NY/2) % NY) - NY/2;
			kj2 = jj*jj+kk2;
			for(i=0; i<d1; i++){
				ii = ((i+NX/2) % NX) - NX/2;
				twiddle[k][j][i] = exp(ap*(double)(ii*ii+kj2));
			}
		}
	}
}

static void compute_initial_conditions(void* pointer_u0, int d1, int d2, int d3){
	dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	int k, j;
	double x0, start, an, starts[NZ];
	start = SEED;
	ipow46(A, 0, &an);
	randlc(&start, an);
	ipow46(A, 2*NX*NY, &an);
	starts[0] = start;
	for(int k=1; k<dims[2]; k++){
		randlc(&start, an);
		starts[k] = start;
	}
	for(k=0; k<dims[2]; k++){
		x0 = starts[k];
		for(j=0; j<dims[1]; j++){
			vranlc(2*NX, &x0, A, (double*)&u0[k][j][0]);
		}
	}
}

static void evolve(void* pointer_u0, void* pointer_u1,
		void* pointer_twiddle, int d1, int d2, int d3){
	dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;
	int i, j, k;
	for(k=0; k<d3; k++)
		for(j=0; j<d2; j++)
			for(i=0; i<d1; i++){
				u0[k][j][i] = dcomplex_mul2(u0[k][j][i], twiddle[k][j][i]);
				u1[k][j][i] = u0[k][j][i];
			}
}

static void fft(int dir, void* pointer_x1, void* pointer_x2){
	dcomplex y1[MAXDIM*FFTBLOCKPAD];
	dcomplex y2[MAXDIM*FFTBLOCKPAD];
	if(dir==1){
		cffts1(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts2(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts3(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x2,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
	}else{
		cffts3(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts2(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
		cffts1(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x2,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y1,
				(dcomplex(*)[FFTBLOCKPAD])(void*)y2);
	}
}

static void fft_init(int n){
	int m, ku, i, j, ln;
	double t, ti;
	m = ilog2(n);
	u[0] = dcomplex_create((double)m, 0.0);
	ku = 2;
	ln = 1;
	for(j=1; j<=m; j++){
		t = PI / ln;
		for(i=0; i<=ln-1; i++){
			ti = i * t;
			u[i+ku-1] = dcomplex_create(cos(ti), sin(ti));
		}
		ku = ku + ln;
		ln = 2 * ln;
	}
}

static void fftz2(int is, int l, int m, int n, int ny, int ny1,
		dcomplex u[], dcomplex x[][FFTBLOCKPAD], dcomplex y[][FFTBLOCKPAD]){
	int k, n1, li, lj, lk, ku, i, j, i11, i12, i21, i22;
	dcomplex u1, x11, x21;
	n1 = n / 2;
	lk = 1 << (l - 1);
	li = 1 << (m - l);
	lj = 2 * lk;
	ku = li;
	for(i=0; i<=li-1; i++){
		i11 = i * lk;
		i12 = i11 + n1;
		i21 = i * lj;
		i22 = i21 + lk;
		if(is>=1) u1 = u[ku+i];
		else      u1 = dconjg(u[ku+i]);
		for(k=0; k<=lk-1; k++){
			for(j=0; j<ny; j++){
				x11 = x[i11+k][j];
				x21 = x[i12+k][j];
				y[i21+k][j] = dcomplex_add(x11, x21);
				y[i22+k][j] = dcomplex_mul(u1, dcomplex_sub(x11, x21));
			}
		}
	}
}

static int ilog2(int n){
	int nn, lg;
	if(n==1){return 0;}
	lg = 1;
	nn = 2;
	while(nn<n){nn*=2; lg+=1;}
	return lg;
}

static void init_ui(void* pointer_u0, void* pointer_u1,
		void* pointer_twiddle, int d1, int d2, int d3){
	dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;
	int i, j, k;
	for(k=0; k<d3; k++)
		for(j=0; j<d2; j++)
			for(i=0; i<d1; i++){
				u0[k][j][i] = dcomplex_create(0.0, 0.0);
				u1[k][j][i] = dcomplex_create(0.0, 0.0);
				twiddle[k][j][i] = 0.0;
			}
}

static void ipow46(double a, int exponent, double* result){
	double q, r;
	int n, n2;
	*result = 1;
	if(exponent==0){return;}
	q = a;
	r = 1;
	n = exponent;
	while(n > 1){
		n2 = n/2;
		if(n2*2==n){randlc(&q, q); n = n2;}
		else        {randlc(&r, q); n = n-1;}
	}
	randlc(&r, q);
	*result = r;
}

static void setup(){
	FILE* fp;
	debug = FALSE;
	if((fp = fopen("timer.flag", "r")) != NULL){
		timers_enabled = TRUE;
		fclose(fp);
	}else{
		timers_enabled = FALSE;
	}
	niter = NITER_DEFAULT;
	dims[0] = NX;
	dims[1] = NY;
	dims[2] = NZ;
}
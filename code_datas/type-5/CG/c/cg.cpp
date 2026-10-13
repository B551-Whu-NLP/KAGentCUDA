// #include "../common/npb-CPP.hpp"
#include "../../libs/headers/npb-CPP.hpp"
#include "../npbparams.hpp"
#include <cstring>

#define NZ (NA*(NONZER+1)*(NONZER+1))
#define NAZ (NA*(NONZER+1))
#define T_INIT 0
#define T_BENCH 1
#define T_CONJ_GRAD 2
#define T_LAST 3

#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];
static double aelt[NAZ];
static double a[NZ];
static double x[NA+2];
static double z[NA+2];
static double p[NA+2];
static double q[NA+2];
static double r[NA+2];
#else
static int (*colidx)=(int*)malloc(sizeof(int)*(NZ));
static int (*rowstr)=(int*)malloc(sizeof(int)*(NA+1));
static int (*iv)=(int*)malloc(sizeof(int)*(NA));
static int (*arow)=(int*)malloc(sizeof(int)*(NA));
static int (*acol)=(int*)malloc(sizeof(int)*(NAZ));
static double (*aelt)=(double*)malloc(sizeof(double)*(NAZ));
static double (*a)=(double*)malloc(sizeof(double)*(NZ));
static double (*x)=(double*)malloc(sizeof(double)*(NA+2));
static double (*z)=(double*)malloc(sizeof(double)*(NA+2));
static double (*p)=(double*)malloc(sizeof(double)*(NA+2));
static double (*q)=(double*)malloc(sizeof(double)*(NA+2));
static double (*r)=(double*)malloc(sizeof(double)*(NA+2));
#endif
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
static boolean timeron;

static void conj_grad(int colidx[], int rowstr[], double x[], double z[],
		double a[], double p[], double q[], double r[], double* rnorm);
static int icnvrt(double x, int ipwr2);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[],
		int firstrow, int lastrow, int firstcol, int lastcol,
		int arow[], int acol[][NONZER+1], double aelt[][NONZER+1], int iv[]);
static void sparse(double a[], int colidx[], int rowstr[], int n, int nz,
		int nozer, int arow[], int acol[][NONZER+1], double aelt[][NONZER+1],
		int firstrow, int lastrow, int nzloc[], double rcond, double shift);
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]);
static void vecset(int n, double v[], int iv[], int* nzv, int i, double val);
static void dump_result(const char* filename, double* x, int n, double zeta);

int main(int argc, char **argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
	int i, j, k, it;
	double zeta;
	double rnorm;
	double norm_temp1, norm_temp2;
	double t, mflops;
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

	for(i=0; i<T_LAST; i++){
		timer_clear(i);
	}

	FILE* fp;
	if((fp = fopen("timer.flag", "r")) != NULL){
		timeron = TRUE;
		fclose(fp);
	}else{
		timeron = FALSE;
	}

	timer_start(T_INIT);

	firstrow = 0;
	lastrow  = NA-1;
	firstcol = 0;
	lastcol  = NA-1;

	if     (NA==1400   && NONZER==7  && NITER==15  && SHIFT==10.0)  class_npb='S';
	else if(NA==7000   && NONZER==8  && NITER==15  && SHIFT==12.0)  class_npb='W';
	else if(NA==14000  && NONZER==11 && NITER==15  && SHIFT==20.0)  class_npb='A';
	else if(NA==75000  && NONZER==13 && NITER==75  && SHIFT==60.0)  class_npb='B';
	else if(NA==150000 && NONZER==15 && NITER==75  && SHIFT==110.0) class_npb='C';
	else if(NA==1500000&& NONZER==21 && NITER==100 && SHIFT==500.0) class_npb='D';
	else if(NA==9000000&& NONZER==26 && NITER==100 && SHIFT==1500.0)class_npb='E';
	else                                                              class_npb='U';

	naa = NA;
	nzz = NZ;

	tran  = 314159265.0;
	amult = 1220703125.0;
	zeta  = randlc(&tran, amult);

	makea(naa, nzz, a, colidx, rowstr, firstrow, lastrow, firstcol, lastcol,
			arow, (int(*)[NONZER+1])(void*)acol,
			(double(*)[NONZER+1])(void*)aelt, iv);

	for(j = 0; j < lastrow - firstrow + 1; j++){
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			colidx[k] = colidx[k] - firstcol;
		}
	}

	for(i = 0; i < NA+1; i++) x[i] = 1.0;
	for(j = 0; j < lastcol-firstcol+1; j++){
		q[j] = 0.0; z[j] = 0.0; r[j] = 0.0; p[j] = 0.0;
	}
	zeta = 0.0;

	// warmup
	for(it = 1; it <= 1; it++){
		conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
		norm_temp1 = 0.0; norm_temp2 = 0.0;
		for(j = 0; j < lastcol - firstcol + 1; j++){
			norm_temp1 += x[j] * z[j];
			norm_temp2 += z[j] * z[j];
		}
		norm_temp2 = 1.0 / sqrt(norm_temp2);
		for(j = 0; j < lastcol - firstcol + 1; j++){
			x[j] = norm_temp2 * z[j];
		}
	}

	for(i = 0; i < NA+1; i++) x[i] = 1.0;
	zeta = 0.0;

	timer_stop(T_INIT);
	timer_start(T_BENCH);

	for(it = 1; it <= NITER; it++){
		if(timeron){timer_start(T_CONJ_GRAD);}
		conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
		if(timeron){timer_stop(T_CONJ_GRAD);}

		norm_temp1 = 0.0; norm_temp2 = 0.0;
		for(j = 0; j < lastcol - firstcol + 1; j++){
			norm_temp1 += x[j] * z[j];
			norm_temp2 += z[j] * z[j];
		}
		norm_temp2 = 1.0 / sqrt(norm_temp2);
		zeta = SHIFT + 1.0 / norm_temp1;

		for(j = 0; j < lastcol - firstcol + 1; j++){
			x[j] = norm_temp2 * z[j];
		}
	}

	timer_stop(T_BENCH);
	t = timer_read(T_BENCH);

	// ── 始终只输出总耗时 ──────────────────────────────────
	printf("Time: %.6f(s)\n", t);
	// ─────────────────────────────────────────────────────

	if(t != 0.0){
		mflops = (double)(2.0*NITER*NA)
			* (3.0+(double)(NONZER*(NONZER+1))
				+ 25.0*(5.0+(double)(NONZER*(NONZER+1)))+3.0)
			/ t / 1000000.0;
	}else{
		mflops = 0.0;
	}

	// ── verbose 模式：dump 结果 + 写 cg_out.txt ──────────
	if(verbose){
		dump_result("cg_out.txt", x, lastcol-firstcol+1, zeta);

		FILE* fout = fopen("cg_ref.txt", "w");
		if(fout == NULL){
			fprintf(stderr, "Error: cannot open cg_out.txt for writing\n");
		}else{
			fprintf(fout, "class      = %c\n", class_npb);
			fprintf(fout, "size       = %d\n", NA);
			fprintf(fout, "iterations = %d\n", NITER);
			fprintf(fout, "time(s)    = %.6f\n", t);
			fprintf(fout, "mflops     = %.2f\n", mflops);
			fclose(fout);
		}
	}
	// ─────────────────────────────────────────────────────

	return 0;
}

void dump_result(const char* filename, double* x, int n, double zeta){
	FILE* fp = fopen(filename, "w");
	if(fp == NULL){
		fprintf(stderr, "Error: cannot open %s for writing\n", filename);
		return;
	}
	// 第一行写 zeta 和向量长度
	fprintf(fp, "%.17f\n", zeta);
	fprintf(fp, "%d\n", n);
	for(int i = 0; i < n; i++){
		fprintf(fp, "%.17f\n", x[i]);
	}
	fclose(fp);
}


static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm){
	int j, k;
	int cgit, cgitmax;
	double d, sum, rho, rho0, alpha, beta;

	cgitmax = 25;

	rho = 0.0;

	
	for(j = 0; j < naa+1; j++){
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
	}

	
	for(j = 0; j < lastcol - firstcol + 1; j++){
		rho = rho + r[j]*r[j];
	}

	
	for(cgit = 1; cgit <= cgitmax; cgit++){
		
		for(j = 0; j < lastrow - firstrow + 1; j++){
			sum = 0.0;
			for(k = rowstr[j]; k < rowstr[j+1]; k++){
				sum = sum + a[k]*p[colidx[k]];
			}
			q[j] = sum;
		}

		
		d = 0.0;
		for (j = 0; j < lastcol - firstcol + 1; j++) {
			d = d + p[j]*q[j];
		}

		
		alpha = rho / d;

		
		rho0 = rho;

		
		rho = 0.0;
		for(j = 0; j < lastcol - firstcol + 1; j++){
			z[j] = z[j] + alpha*p[j];
			r[j] = r[j] - alpha*q[j];
		}

		
		for(j = 0; j < lastcol - firstcol + 1; j++){
			rho = rho + r[j]*r[j];
		}

		
		beta = rho / rho0;

		
		for(j = 0; j < lastcol - firstcol + 1; j++){
			p[j] = r[j] + beta*p[j];
		}
	} 

	
	sum = 0.0;
	for(j = 0; j < lastrow - firstrow + 1; j++){
		d = 0.0;
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			d = d + a[k]*z[colidx[k]];
		}
		r[j] = d;
	}

	
	for(j = 0; j < lastcol-firstcol+1; j++){
		d   = x[j] - r[j];
		sum = sum + d*d;
	}

	*rnorm = sqrt(sum);
}


static int icnvrt(double x, int ipwr2){
	return (int)(ipwr2 * x);
}


static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]){
	int iouter, ivelt, nzv, nn1;
	int ivc[NONZER+1];
	double vc[NONZER+1];

	
	nn1 = 1;
	do{
		nn1 = 2 * nn1;
	}while(nn1 < n);

	
	for(iouter = 0; iouter < n; iouter++){
		nzv = NONZER;
		sprnvc(n, nzv, nn1, vc, ivc);
		vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
		arow[iouter] = nzv;
		for(ivelt = 0; ivelt < nzv; ivelt++){
			acol[iouter][ivelt] = ivc[ivelt] - 1;
			aelt[iouter][ivelt] = vc[ivelt];
		}
	}

	
	sparse(a,
			colidx,
			rowstr,
			n,
			nz,
			NONZER,
			arow,
			acol,
			aelt,
			firstrow,
			lastrow,
			iv,
			RCOND,
			SHIFT);
}


static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift){	
	int nrows;

	
	int i, j, j1, j2, nza, k, kk, nzrow, jcol;
	double size, scale, ratio, va;
	boolean goto_40;

	
	nrows = lastrow - firstrow + 1;

	
	for(j = 0; j < nrows+1; j++){
		rowstr[j] = 0;
	}
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza] + 1;
			rowstr[j] = rowstr[j] + arow[i];
		}
	}
	rowstr[0] = 0;
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] + rowstr[j-1];
	}
	nza = rowstr[nrows] - 1;

	
	if(nza > nz){
		printf("Space for matrix elements exceeded in sparse\n");
		printf("nza, nzmax = %d, %d\n", nza, nz);
		exit(EXIT_FAILURE);
	}

	
	for(j = 0; j < nrows; j++){
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			a[k] = 0.0;
			colidx[k] = -1;
		}
		nzloc[j] = 0;
	}

	
	size = 1.0;
	ratio = pow(rcond, (1.0 / (double)(n)));
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza];

			scale = size * aelt[i][nza];
			for(nzrow = 0; nzrow < arow[i]; nzrow++){
				jcol = acol[i][nzrow];
				va = aelt[i][nzrow] * scale;

				
				if(jcol == j && j == i){
					va = va + rcond - shift;
				}

				goto_40 = FALSE;
				for(k = rowstr[j]; k < rowstr[j+1]; k++){
					if(colidx[k] > jcol){
						
						for(kk = rowstr[j+1]-2; kk >= k; kk--){
							if(colidx[kk] > -1){
								a[kk+1] = a[kk];
								colidx[kk+1] = colidx[kk];
							}
						}
						colidx[k] = jcol;
						a[k]  = 0.0;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == -1){
						colidx[k] = jcol;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == jcol){
						
						nzloc[j] = nzloc[j] + 1;
						goto_40 = TRUE;
						break;
					}
				}
				if(goto_40 == FALSE){
					printf("internal error in sparse: i=%d\n", i);
					exit(EXIT_FAILURE);
				}
				a[k] = a[k] + va;
			}
		}
		size = size * ratio;
	}

	
	for(j = 1; j < nrows; j++){
		nzloc[j] = nzloc[j] + nzloc[j-1];
	}

	for(j = 0; j < nrows; j++){
		if(j > 0){
			j1 = rowstr[j] - nzloc[j-1];
		}else{
			j1 = 0;
		}
		j2 = rowstr[j+1] - nzloc[j];
		nza = rowstr[j];
		for(k = j1; k < j2; k++){
			a[k] = a[nza];
			colidx[k] = colidx[nza];
			nza = nza + 1;
		}
	}
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] - nzloc[j-1];
	}
	nza = rowstr[nrows] - 1;
}


static void sprnvc(int n, int nz, int nn1, double v[], int iv[]){
	int nzv, ii, i;
	double vecelt, vecloc;

	nzv = 0;

	while(nzv < nz){
		vecelt = randlc(&tran, amult);

		
		vecloc = randlc(&tran, amult);
		i = icnvrt(vecloc, nn1) + 1;
		if(i>n){continue;}

		
		boolean was_gen = FALSE;
		for(ii = 0; ii < nzv; ii++){
			if(iv[ii] == i){
				was_gen = TRUE;
				break;
			}
		}
		if(was_gen){continue;}
		v[nzv] = vecelt;
		iv[nzv] = i;
		nzv = nzv + 1;
	}
}


static void vecset(int n, double v[], int iv[], int* nzv, int i, double val){
	int k;
	boolean set;

	set = FALSE;
	for(k = 0; k < *nzv; k++){
		if(iv[k] == i){
			v[k] = val;
			set  = TRUE;
		}
	}
	if(set == FALSE){
		v[*nzv]  = val;
		iv[*nzv] = i;
		*nzv     = *nzv + 1;
	}
}

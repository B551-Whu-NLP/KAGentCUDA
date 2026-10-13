#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void set_sg(int *sxz, int sxbeg, int szbeg, int jsx, int jsz, int ns, int npml, int nnz) {
    for (int id = 0; id < ns; id++) {
        sxz[id] = nnz * (sxbeg + id * jsx + npml) + (szbeg + id * jsz + npml);
    }
}

int main(int argc, char *argv[]) {
    const int ns = 10000000; // 增加数组大小
    int sxbeg = 0;
    int szbeg = 0;
    int jsx = 1;
    int jsz = 1;
    int npml = 0;
    int nnz = 5;
    int* h_sxz = (int*)malloc(ns * sizeof(int));
    
    clock_t start = clock();
    
    set_sg(h_sxz, sxbeg, szbeg, jsx, jsz, ns, npml, nnz);
    
    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_76_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_sxz[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_sxz);
    printf("Time: %.2f(ms)\n", time_spent * 1000);
    return 0;
}
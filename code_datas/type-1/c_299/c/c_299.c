#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void check_results(int* g_results0, int* g_results1, int n){
    unsigned int gidx;
    unsigned int result0;
    unsigned int result1;
    for (gidx = 0; gidx < n; gidx++) {
        result0 = g_results0[gidx];
        result1 = g_results1[gidx];
        if (result0 != result1) {
            printf("%d != %d for thread %d\n", result0, result1, gidx);
        }
    }
}

int main(int argc, char *argv[]) {
    const int n = 1000000; // 增大数组大小
    int* h_results0 = (int*)malloc(n * sizeof(int));
    int* h_results1 = (int*)malloc(n * sizeof(int));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start_time = clock();

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_results0[i] = rand() % 100;
        h_results1[i] = h_results0[i] * 2;
    }

    // 增加额外的计算循环来延长执行时间
    for (int iter = 0; iter < 50000; iter++) {
        for (int i = 0; i < n; i += 1000) {
            h_results0[i] += iter % 10;
        }
    }

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    check_results(h_results0, h_results1, n);

    if (verbose) {
        FILE *fp = fopen("c_299_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d %d\n", h_results0[i], h_results1[i]);
            }
            fclose(fp);
        }
    }

    free(h_results0);
    free(h_results1);

    return 0;
}
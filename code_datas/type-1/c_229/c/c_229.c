#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void transferMBR3(double* xy_copy, long long* a_copy, int tasks);

int main(int argc, char *argv[]) {
    int tasks = 100000000; // 增加到1亿
    double* h_xy_copy;
    long long* h_a_copy;
    h_xy_copy = (double*)malloc(tasks * sizeof(double));
    h_a_copy = (long long*)malloc(tasks * sizeof(long long));

    for (int i = 0; i < tasks; ++i) {
        h_xy_copy[i] = i * 1.5;
    }

    clock_t start = clock();

    transferMBR3(h_xy_copy, h_a_copy, tasks);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_229_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%lld ", h_a_copy[i]);
            }
            fclose(fp);
        }
    }

    free(h_xy_copy);
    free(h_a_copy);

    return 0;
}

void transferMBR3(double* xy_copy, long long* a_copy, int tasks) {
    for (int i = 0; i < tasks; i++) {
        a_copy[i] = xy_copy[i] * 10000000;
    }
}
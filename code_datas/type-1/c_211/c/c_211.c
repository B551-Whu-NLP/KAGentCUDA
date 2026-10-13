#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void addV(int* a, int* b, int* c, int N) {
    for (int index = 0; index < N; index++) {
        c[index] = a[index] + b[index];
    }
}

int main(int argc, char* argv[]) {
    const int array_size = 10000000; // 增大数组大小
    int* a_host = (int*)malloc(array_size * sizeof(int));
    int* b_host = (int*)malloc(array_size * sizeof(int));
    int* c_result = (int*)malloc(array_size * sizeof(int));

    if (!a_host || !b_host || !c_result) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < array_size; ++i) {
        a_host[i] = i + 1;
        b_host[i] = (i + 1) * 2;
    }

    clock_t start = clock();

    addV(a_host, b_host, c_result, array_size);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* fp = fopen("c_211_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", c_result[i]);
            }
            fclose(fp);
        }
    } 

    free(a_host);
    free(b_host);
    free(c_result);

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    return 0;
}
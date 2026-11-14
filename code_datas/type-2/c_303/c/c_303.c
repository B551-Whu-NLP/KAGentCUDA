#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void kernel(int* a, int* b, int* c, int size) {
    for (int idx = 0; idx < size; idx++) {
        int idx1 = (idx + 1) % size;
        int idx2 = (idx + 2) % size;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(int argc, char** argv) {
    int size = 1000000; // 增大数组大小
    int* a = (int*)malloc(size * sizeof(int));
    int* b = (int*)malloc(size * sizeof(int));
    int* c = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = size - i;
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 1000; iter++) {
        kernel(a, b, c, size);
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_303_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", c[i]);
            }
            fclose(file);
        }
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
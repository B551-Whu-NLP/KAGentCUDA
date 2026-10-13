#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void kernel(int* a, int* b, int* c, int size) {
    for (int idx = 0; idx < size; idx++) {
        int idx1 = (idx + 1) % 4;
        int idx2 = (idx + 2) % 4;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(int argc, char** argv) {
    int size = 1024 * 10000; // 进一步增大数组大小
    int* a = (int*)malloc(size * sizeof(int));
    int* b = (int*)malloc(size * sizeof(int));
    int* c = (int*)malloc(size * sizeof(int));
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = size - i;
    }

    clock_t start = clock();
    kernel(a, b, c, size);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_302_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < size; i++) {
                fprintf(fp, "%d ", c[i]);
            }
            fclose(fp);
        }
    }

    free(a);
    free(b);
    free(c);
    return 0;
}
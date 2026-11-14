#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mathKernel1(float* c, int size) {
    int tid;
    for (tid = 0; tid < size; tid++) {
        float ia, ib;
        ia = ib = 0.0f;
        if (tid % 2 == 0) {
            ia = 100.0f;
        }
        else {
            ib = 200.0f;
        }
        c[tid] = ia + ib;
    }
}

int main(int argc, char* argv[]) {
    int size = 100000000; // 增加100倍
    float* h_c = (float*)malloc(size * sizeof(float));
    
    srand(42);
    for(int i = 0; i < size; i++) {
        h_c[i] = (float)(rand() % 1000) / 10.0f;
    }

    clock_t start = clock();
    mathKernel1(h_c, size);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_247_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(f, "%.2f ", h_c[i]);
            }
            fclose(f);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    free(h_c);
    return 0;
}
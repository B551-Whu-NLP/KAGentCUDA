#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void pythagoras(unsigned char* a, unsigned char* b, unsigned char* c, int size) {
    for (int idx = 0; idx < size; idx++) {
        float af = (float)(a[idx]);
        float bf = (float)(b[idx]);
        c[idx] = (unsigned char)sqrtf(af * af + bf * bf);
    }
}

int main(int argc, char *argv[]) {
    int size = 1000;
    unsigned char* h_a = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_b = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_c = (unsigned char*)malloc(size * sizeof(unsigned char));

    for (int i = 0; i < size; ++i) {
        h_a[i] = 100;
        h_b[i] = 150;
    }

    clock_t start = clock();

    pythagoras(h_a, h_b, h_c, size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_273_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%u ", h_c[i]);
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void softmax(float *x, int r, int c) {
    float temp1, temp2;
    for (int i = 0; i < r; i++) {
        temp1 = 0.;
        temp2 = 0.;
        for (int j = 0; j < c; j++) {
            temp1 = fmaxf(x[i * c + j], temp1);
        }
        for (int j = 0; j < c; j++) {
            x[i * c + j] = expf(x[i * c + j] - temp1);
            temp2 += x[i * c + j];
        }
        for (int j = 0; j < c; j++) {
            x[i * c + j] /= temp2;
        }
    }
}

int main(int argc, char *argv[]) {
    int r = 10000;
    int c = 20000;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float *x = (float *)malloc(r * c * sizeof(float));

    for (int i = 0; i < r * c; i++) {
        x[i] = i % 100 + 1; 
    }

    clock_t start = clock();
    softmax(x, r, c);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);


    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_327_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.4f\t", x[i * c + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(x);

    return 0;
}
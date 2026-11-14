#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void diffusion(double* x0, double* x1, int nx, int ny, double dt, int verbose) {
    int i, j;
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (j = 1; j < ny - 1; ++j) { 
            for (i = 1; i < nx - 1; ++i) {
                int pos = nx * j + i;
                x1[pos] = x0[pos] + dt * (-4.0 * x0[pos] + x0[pos - 1] + x0[pos + 1] + x0[pos - nx] + x0[pos + nx]);
            }
        }
        // Swap pointers for next iteration
        double* temp = x0;
        x0 = x1;
        x1 = temp;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_121_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int k = 0; k < 1000; k++) {
                fprintf(fp, "h_x1[%d] = %f\n", k, x1[k]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    int nx = 1000;
    int ny = 1000;
    double dt = 0.01;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    double* h_x0 = (double*)malloc(nx * ny * sizeof(double));
    double* h_x1 = (double*)malloc(nx * ny * sizeof(double));

    for (int i = 0; i < nx * ny; i++) {
        h_x0[i] = 1.0;  
        h_x1[i] = 0;
    }

    diffusion(h_x0, h_x1, nx, ny, dt, verbose);

    free(h_x0);
    free(h_x1);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void getRho(const int numOfNucl, const double* psi, const double* occNo, double* rho) {
    *rho = 0;

    for (int i = 0; i < numOfNucl; ++i)
        *rho += occNo[i] * psi[i];

}

int main(int argc, char** argv) {
    int array_size = 100000000;

    double* h_psi = (double*)malloc(array_size * sizeof(double));
    double* h_occNo = (double*)malloc(array_size * sizeof(double));
    double* h_rho = (double*)malloc(sizeof(double));

    for (int i = 0; i < array_size; i++) {
        h_psi[i] = 1.0;
        h_occNo[i] = 2.0;
    }

    clock_t start_time = clock();

    getRho(array_size, h_psi, h_occNo, h_rho);

    clock_t end_time = clock();

    double time_in_seconds = (end_time - start_time) / (double)CLOCKS_PER_SEC;
    double time_in_ms = time_in_seconds * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    printf("Time: %.2f(ms)\n", time_in_ms);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_129_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            fprintf(fp, "Result rho: %f\n", *h_rho);
            fclose(fp);
        }
    }

    free(h_psi);
    free(h_occNo);
    free(h_rho);
    return 0;
}
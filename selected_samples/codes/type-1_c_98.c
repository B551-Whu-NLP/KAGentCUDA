#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void envejecer_kernel(int* estado, int* edad, int* pupacion, int N, int dia) {
    clock_t start = clock();

    for (int iter = 0; iter < 10000; iter++) {
        for (int id = 0; id < N; id++) {
            if (dia < 80 || dia > 320) {
                if (edad[id] > pupacion[id]) {
                    edad[id]++;
                }
            }
            else {
                edad[id]++;
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char* argv[]) {
    int N = 100000;
    int dia = 100;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int* h_estado = (int*)malloc(N * sizeof(int));
    int* h_edad = (int*)malloc(N * sizeof(int));
    int* h_pupacion = (int*)malloc(N * sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) {
        h_edad[i] = 0;
        h_pupacion[i] = 10 + (i % 20);
        h_estado[i] = 0;
    }

    envejecer_kernel(h_estado, h_edad, h_pupacion, N, dia);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_98_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", h_edad[i]);
                if ((i + 1) % 10 == 0) fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_estado);
    free(h_edad);
    free(h_pupacion);

    return 0;
}
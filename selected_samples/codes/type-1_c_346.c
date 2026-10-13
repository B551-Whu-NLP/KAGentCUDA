#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double RyT(float* R, float* T, float* P, float* Q, int num_points) {
    clock_t start = clock();

    for (int i = 0; i < num_points; i++) {
        Q[0 + i * 3] = R[0 + 0 * 3] * P[0 + i * 3] + R[0 + 1 * 3] * P[1 + i * 3] +
            R[0 + 2 * 3] * P[2 + i * 3] + T[0];
        Q[1 + i * 3] = R[1 + 0 * 3] * P[0 + i * 3] + R[1 + 1 * 3] * P[1 + i * 3] +
            R[1 + 2 * 3] * P[2 + i * 3] + T[1];
        Q[2 + i * 3] = R[2 + 0 * 3] * P[0 + i * 3] + R[2 + 1 * 3] * P[1 + i * 3] +
            R[2 + 2 * 3] * P[2 + i * 3] + T[2];
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[]) {
    const int num_points = 1024 * 1000 * 10; // 进一步增大点数量以增加计算量
    float* R, * T, * P, * Q;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    R = (float*)malloc(9 * sizeof(float));
    T = (float*)malloc(3 * sizeof(float));
    P = (float*)malloc(3 * num_points * sizeof(float));
    Q = (float*)malloc(3 * num_points * sizeof(float));

    R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
    R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
    R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;

    T[0] = 1.0; T[1] = 2.0; T[2] = 3.0;

    for (int i = 0; i < num_points * 3; ++i) {
        P[i] = (float)(i);
    }

    double time_spent = RyT(R, T, P, Q, num_points);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_346_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "Q[%d]: [%f, %f, %f]\n", i, Q[i * 3], Q[i * 3 + 1], Q[i * 3 + 2]);
            }
            fclose(file);
        }
    }

    free(R);
    free(T);
    free(P);
    free(Q);
    
    printf("Time: %.2f(ms)\n", time_spent * 1000);
    return 0;
}
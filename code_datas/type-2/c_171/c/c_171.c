#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void* Match(int num_points, float* P, float* Q, int q_points, int* idx, int start, int end) {
    float dist;
    float max_dist;

    clock_t start_time = clock();

    for (int i = start; i < end; i++) {
        max_dist = 100.0f;

        for (int j = 0; j < num_points; j++) {
            dist = (P[0 + i * 3] - Q[0 + j * 3]) * (P[0 + i * 3] - Q[0 + j * 3]) +
                (P[1 + i * 3] - Q[1 + j * 3]) * (P[1 + i * 3] - Q[1 + j * 3]) +
                (P[2 + i * 3] - Q[2 + j * 3]) * (P[2 + i * 3] - Q[2 + j * 3]);

            if (dist < max_dist) {
                max_dist = dist;
                idx[i] = j;
            }
        }
    }

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken);

    return NULL;
}

int main(int argc, char *argv[]) {
    const int num_points = 50000;
    const int q_points = 50000;
    const int points_per_coordinate = 3;
    const int array_size = num_points * points_per_coordinate;
    const int start = 0;
    const int end = num_points;

    float *P = (float*)malloc(array_size * sizeof(float));
    float *Q = (float*)malloc(array_size * sizeof(float));

    int *idx = (int*)malloc(num_points * sizeof(int)); 

    for (int i = 0; i < array_size; ++i) {
        P[i] = (i + 1) * 0.1f;
    }
    for (int i = 0; i < array_size; ++i) {
        Q[i] = (i + 10) * 0.1f;
    }
    for (int i = 0; i < num_points; ++i) {
        idx[i] = 0;
    }

    Match(num_points, P, Q, q_points, idx, start, end);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_171_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d %d\n", i, idx[i]);
            }
            fclose(fp);
        }
    }

    free(P);
    free(Q);
    free(idx);

    return 0;
}
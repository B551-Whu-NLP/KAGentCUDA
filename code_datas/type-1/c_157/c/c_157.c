#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void k_adam_kernel(float* m, float* v, float* w, const float* d, int max_size, float beta1, float beta2, float beta1_tpower, float beta2_tpower, float learning_rate) {
    const float eps = 1e-8;

    clock_t start = clock();

    for (int i = 0; i < max_size; i++) {
        float d_temp = d[i];
        m[i] = m[i] * beta1 + d_temp * (1 - beta1);
        v[i] = v[i] * beta2 + d_temp * d_temp * (1 - beta2);

        float m_hat = m[i] / (1 - beta1_tpower);
        float v_hat = sqrtf(v[i] / (1 - beta2_tpower)) + eps;
        
        w[i] += (m_hat / v_hat) * (-learning_rate);
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f\n", time_spent);
}

int main(int argc, char *argv[]) {
    int max_size = 1000;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float beta1_tpower = 0.5;
    float beta2_tpower = 0.5;
    float learning_rate = 0.001;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* m = (float*)malloc(max_size * sizeof(float));
    float* v = (float*)malloc(max_size * sizeof(float));
    float* w = (float*)malloc(max_size * sizeof(float));
    float* d = (float*)malloc(max_size * sizeof(float));

    for (int i = 0; i < max_size; i++) {
        m[i] = i + 1;
        v[i] = i + 2;
        w[i] = i + 3;
        d[i] = i + 4;
    }

    k_adam_kernel(m, v, w, d, max_size, beta1, beta2, beta1_tpower, beta2_tpower, learning_rate);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_157_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < max_size; i++) {
                fprintf(fp, "%.6f ", w[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    free(m);
    free(v);
    free(w);
    free(d);
    return 0;
}
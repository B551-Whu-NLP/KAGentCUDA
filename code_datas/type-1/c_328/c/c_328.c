#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void cuda_Adam_step_kernel(float *grad, float *data, float *m, float *v,
                            short decay, float weight_decay,
                            float beta1, float beta2,
                            float eps, float step_size, int varsize) {
    for (int i = 0; i < varsize; i++) {
        float g = grad[i];
        if (decay) 
            g += weight_decay * data[i];

        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        data[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
    }
}

int main(int argc, char *argv[]) {
    int varsize = 1000; 

    float *grad = (float *)malloc(varsize * sizeof(float));
    float *data = (float *)malloc(varsize * sizeof(float));
    float *m = (float *)malloc(varsize * sizeof(float));
    float *v = (float *)malloc(varsize * sizeof(float));

    for (int i = 0; i < varsize; i++) {
        grad[i] = i + 1; 
        data[i] = i + 2;
        m[i] = 2 * i; 
        v[i] = 2 * i + 1;
    }

    short decay = 1;
    float weight_decay = 0.01; 

    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    float step_size = 0.001;

    clock_t start = clock();

    cuda_Adam_step_kernel(grad, data, m, v, decay, weight_decay, beta1, beta2, eps, step_size, varsize);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_328_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < varsize; i++) {
                fprintf(fp, "%.4f ", data[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(grad);
    printf("Time: %.2f(ms)\n", time_taken);
    free(data);
    free(m);
    free(v);

    return 0;
}
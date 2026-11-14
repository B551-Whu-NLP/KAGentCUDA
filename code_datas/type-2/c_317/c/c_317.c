#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void backward_avgpool_layer(int batch, int c, int w, int h, float* delta) {
    clock_t start = clock();

    int id, i, k;

    for (id = 0; id < batch; ++id) {
        for (k = 0; k < c; ++k) {
            int out_index = k + id * c;

            for (i = 0; i < h * w; ++i) {
                int in_index = i + h * w * (k + id * c);
                delta[in_index] += delta[out_index] / (h * w);
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int batch = 100; 
    int c = 160;
    int w = 320; 
    int h = 320;

    float *delta = (float *)malloc(batch * c * w * h * sizeof(float));

    for (int i = 0; i < batch * c * w * h; i++) {
        delta[i] = i + 1.0; 
    }

    backward_avgpool_layer(batch, c, w, h, delta);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_317_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
           for (int j = 0; j < batch * c; j++) {
                for (int i = 0; i < 5; i++) {
                    int index = i + j * (w * h);
                    fprintf(fp, "%6.2f ", delta[index]);
                }
            }
        }
    }

    free(delta);
    return 0;
}
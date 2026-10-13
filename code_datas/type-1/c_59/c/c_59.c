#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void forward_dropout_layer(int size, float *input, float prob, float *rand, float scale) {
    for (int i = 0; i < size; ++i) {
        if (rand[i] < prob) {
            input[i] = 0;
        } else {
            input[i] *= scale;
        }
    }
}

int main(int argc, char *argv[]) {
    const int size = 50000000; // 增大数组大小
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_rand = (float*)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; ++i) {
        h_input[i] = (float)(i);
        h_rand[i] = (float)(i) / size; 
    }

    clock_t start = clock();
    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 100; ++iter) {
        forward_dropout_layer(size, h_input, 0.5f , h_rand, 2.0f);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_59_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_input[%d]: %f\n", i, h_input[i]);
            }
            fclose(fp);
        }
    }

    printf("Time: %.2f(ms)\n", time_taken);
    
    free(h_input);
    free(h_rand);
    return 0;
}
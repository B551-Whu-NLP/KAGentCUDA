#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Function_update(float lr, float* parameter, float* gradient, int size) {
    for (int i = 0; i < size; i++) {
        parameter[i] -= lr * gradient[i];
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 1000000;
    float* h_dev_parameter = (float*)malloc(arraySize * sizeof(float));
    float* h_dev_gradient = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_dev_parameter[i] = (float)(i);
        h_dev_gradient[i] = (float)(2 * i);
    }

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        Function_update(0.01f, h_dev_parameter, h_dev_gradient, arraySize);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_42_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%f ", h_dev_gradient[i]);
            }
            fclose(file);
        }
    }

    free(h_dev_parameter);
    free(h_dev_gradient);
    return 0;
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void Gather(const int* input, float* output, int input_size, const float* data, int count, int dim, int data_offset) {
    int index;
    for (index = 0; index < input_size * dim; index++) {
        output[index] = 0; 
        const int input_id = input[index / dim];
        const int pos = index % dim;
        if (input_id < count + data_offset && input_id >= data_offset) {
            output[index] = data[input_id * dim + pos];
        }
    }
}

int main(int argc, char* argv[]) {
    const int input_size = 100000;
    const int count = 100000;
    const int dim = 10;
    const int data_offset = 5000;
    int* h_input = (int*)malloc(input_size * sizeof(int));
    float* h_output = (float*)malloc(input_size * dim * sizeof(float));
    float* h_data = (float*)malloc((count + data_offset) * dim * sizeof(float));

    for (int i = 0; i < input_size; ++i) {
        h_input[i] = i % (count + data_offset);
    }
    for (int i = 0; i < (count + data_offset) * dim; ++i) {
        h_data[i] = i;
    }

    clock_t start = clock();

    Gather(h_input, h_output, input_size, h_data, count, dim, data_offset);

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_291_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "Input ID %d: ", h_input[i]);
                for (int j = 0; j < dim; ++j) {
                    fprintf(file, "%.2f ", h_output[i * dim + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_input);
    free(h_output);
    free(h_data);

    return 0;
}
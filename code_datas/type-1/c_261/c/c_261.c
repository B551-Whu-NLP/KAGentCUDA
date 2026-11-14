#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void ReLU_forward(float* in, int* mask, int datasize, int training) {
    for (int i = 0; i < datasize; ++i) {
        int keep = in[i] > 0;
        if (training) {
            mask[i] = keep;
        }
        if (!keep) {
            in[i] = 0;
        }
    }
}

int main(int argc, char** argv) {
    long unsigned int datasize = 10000000; // 增大数据规模
    int training = 1; 
    float* h_d_in_data = (float*)malloc(datasize * sizeof(float));
    int* h_d_mask = (int*)malloc(datasize * sizeof(int)); 

    srand(42);
    for (int i = 0; i < datasize; ++i) {
        h_d_in_data[i] = rand() % 100 - 50;
    }

    clock_t start = clock();
    ReLU_forward(h_d_in_data, h_d_mask, datasize, training);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_261_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_d_in_data[i]);
            }
            fprintf(file, "\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_d_mask[i] ? 1 : 0);
            }
            fclose(file);
        }
    }


    free(h_d_in_data);
    free(h_d_mask);

    printf("Time: %.2fms\n", time_taken * 1000);

    return 0;
}
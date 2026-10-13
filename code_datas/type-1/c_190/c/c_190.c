#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void subAvg(int* input, int count, int avg) {
    for (int index = 0; index < count; index++) {
        input[index] = input[index] - avg;
    }
}

int main(int argc, char* argv[]) {
    const int data_size = 100000000; // Increased from 1000000 to 100000000
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int* input_host = (int*)malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; ++i) {
        input_host[i] = i + 1;
    }

    clock_t start_time = clock();

    int sum = 0;
    for (int i = 0; i < data_size; ++i) {
        sum += input_host[i];
    }
    int avg = sum / data_size;

    subAvg(input_host, data_size, avg);

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time: %.2f ms\n", elapsed_time * 1000);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "c_190_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", input_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(input_host);

    return 0;
}
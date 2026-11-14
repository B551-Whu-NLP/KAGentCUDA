#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void flip(float* array1, int width) {
    clock_t start = clock();

    for (int current_index = 0; current_index < width * width / 2; current_index++) {
        int replace = (width - 1 - current_index / width) * width + current_index % width;
        float temp = array1[current_index];
        array1[current_index] = array1[replace];
        array1[replace] = temp;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char* argv[]) {
    int width = 1000;
    float* h_array1 = (float*)malloc(width * width * sizeof(float));

    for (int i = 0; i < width * width; ++i) {
        h_array1[i] = i + 1.0;
    }

    flip(h_array1, width);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_267_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_array1[i]);
                if ((i + 1) % width == 0) {
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        }
    }

    free(h_array1);
    return 0;
}
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void InitCCL(int labelList[], int reference[], int width, int height) {
    int x, y;
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            int id = x + y * width;
            labelList[id] = reference[id] = id;
        }
    }
}

int main(int argc, char *argv[]) {
    int width = 1000;
    int height = 1000;
    int* h_labelList = (int*)malloc(width * height * sizeof(int));
    int* h_reference = (int*)malloc(width * height * sizeof(int));

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        InitCCL(h_labelList, h_reference, width, height);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_75_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(fp, "%d ", h_labelList[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(fp, "%d ", h_reference[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_labelList);
    free(h_reference);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void bubbleSort(int* d_array, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (d_array[j] > d_array[j + 1]) {
                int temp = d_array[j];
                d_array[j] = d_array[j + 1];
                d_array[j + 1] = temp;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int size = 1000;
    int* h_array = (int*)malloc(size * sizeof(int));
    srand(42);
    for (int i = 0; i < size; i++) {
        h_array[i] = rand() % 1000;
    }
    clock_t start = clock();
    bubbleSort(h_array, size);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_93_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < size; i++) {
                fprintf(fp, "%d ", h_array[i]);
            }
            fclose(fp);
        }
    }
    free(h_array);
    return 0;
}
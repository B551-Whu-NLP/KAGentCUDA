#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_100(int numElements, int* data) {
    for (int idx = 0; idx < numElements; idx++) {
        data[idx] += 100;
    }
}

int main(int argc, char* argv[]) {
    int numElements = 100000000;
    int* h_data = (int*)malloc(numElements * sizeof(int));
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = i;
    }

    clock_t start = clock();
    add_100(numElements, h_data);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_1_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            int print_n = 1000;  // 只输出前1000个
            for (int i = 0; i < print_n; i++) {
                fprintf(file, "%d ", h_data[i]);
            }
            fclose(file);
        }
    }

    free(h_data);
    return 0;
}
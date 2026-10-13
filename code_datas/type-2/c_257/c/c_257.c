#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void HammingDistance(int* c, const int* a, const int* b, const long int size) {
    for (long int i = 0; i < size; i += 1) {
        if (a[i] != b[i])
            (*c)++;
    }
}

int main(int argc, char** argv) {
    long int size = 100000000; // 增大数组大小
    int* h_a = (int*)malloc(size * sizeof(int));
    int* h_b = (int*)malloc(size * sizeof(int));
    int h_c = 0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < size; ++i) {
        h_a[i] = rand() % 2;
        h_b[i] = rand() % 2;
    }

    clock_t start = clock();

    HammingDistance(&h_c, h_a, h_b, size);  

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);
    printf("Hamming Distance: %d\n", h_c);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_257_output.txt");
        FILE* fp = fopen(output_filename, "w");
        if (fp != NULL) {
            fprintf(fp, "Hamming Distance: %d\n", h_c);
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);

    return 0;
}
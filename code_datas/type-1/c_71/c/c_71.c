#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void kmeans_average(int *means, int *counts, int BID, int DIM) {
    clock_t start = clock();

    for (int iter = 0; iter < 10000; iter++) {
        for (int bid = 0; bid < BID; bid++) {
            for (int tid = 0; tid < DIM; tid++) {
                if (counts[bid] == 0) {
                    means[bid * DIM + tid] = 0;
                } 
                else {
                    means[bid * DIM + tid] /= counts[bid];
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char *argv[]) {
    int BID = 300;
    int DIM = 400;
    int* h_means = (int*)malloc(BID * DIM * sizeof(int));
    int* h_counts = (int*)malloc(BID * sizeof(int));
   
    srand(42);
    for (int i = 0; i < BID * DIM; ++i) {
        h_means[i] = rand() % 100;
    }
    for (int i = 0; i < BID; ++i) {
        h_counts[i] = rand() % 10 + 1;
    }
 
    kmeans_average(h_means, h_counts, BID, DIM);
 
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_71_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_means[i]);
            }
            fclose(fp);
        }
    }

    free(h_means);
    free(h_counts);

    return 0;
}
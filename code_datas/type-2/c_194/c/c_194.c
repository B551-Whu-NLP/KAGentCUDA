#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void histogram(int n, int* color, int* bucket) {
    for (int i = 0; i < n; i++) {
        int c = color[i];
        bucket[c] += 1;
    }
}

int main(int argc, char* argv[]) {
    const int data_size = 100000000;  // 增加数据量
    const int num_bins = 4;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int* color_host = (int*)malloc(data_size * sizeof(int));
    int* bucket_host = (int*)malloc(num_bins * sizeof(int));

    srand(42);
    for (int i = 0; i < data_size; i++) {
        color_host[i] = rand() % num_bins;
    }
    memset(bucket_host, 0, num_bins * sizeof(int));

    clock_t start = clock();

    histogram(data_size, color_host, bucket_host);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_194_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < num_bins; i++) {
                fprintf(fp, "%d: %d\n", i, bucket_host[i]);
            }
            fclose(fp);
        }
    }

    free(color_host);
    free(bucket_host);

    return 0;
}
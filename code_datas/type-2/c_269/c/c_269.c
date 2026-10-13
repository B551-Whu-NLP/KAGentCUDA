#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int size = 10000000; // 进一步增大数组大小
    int key = 1500000;   // 调整查找键值
    int* h_gpu_key_arr = (int*)malloc(size * sizeof(int));
    int* h_gpu_pos = (int*)malloc(sizeof(int));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < size; ++i) {
        h_gpu_key_arr[i] = i * 10;
    }

    clock_t start = clock();

    *h_gpu_pos = -1; 
    for (int globalTx = 0; globalTx < size - 1; globalTx++) {
        if (key >= h_gpu_key_arr[globalTx] && key < h_gpu_key_arr[globalTx + 1]) {
            *h_gpu_pos = globalTx;
            break;
        }
    }

    if (key >= h_gpu_key_arr[size - 1]) {
        *h_gpu_pos = size - 1;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "c_269_output.txt");

    if (verbose) {
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_gpu_key_arr[i]);
            }
            fprintf(fp, "\nSearch Key: %d\n", key);
            fprintf(fp, "Position: %d\n", *h_gpu_pos);
            fclose(fp);
        }
    }

    free(h_gpu_key_arr);
    free(h_gpu_pos);

    return 0;
}
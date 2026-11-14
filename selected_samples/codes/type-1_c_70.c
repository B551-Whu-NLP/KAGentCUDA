#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void is_repeat(int N, int *device_input, int *device_output) {
    for (int idx = 0; idx < N; idx++) {
        device_output[idx] = 0;
        if (idx + 1 < N && device_input[idx] == device_input[idx + 1]) {
            device_output[idx] = 1;
        }
    }
}

int main(int argc, char *argv[]) {
    const int N = 1000000; // 增大数组大小
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int* h_device_input = (int*)malloc(N * sizeof(int));
    int* h_device_output = (int*)malloc(N * sizeof(int));

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_device_input[i] = rand() % 10;
    }

    clock_t start_time = clock();
    
    is_repeat(N, h_device_input, h_device_output);
    
    clock_t end_time = clock();
    
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", "c_70");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_device_output[%d]: %d\n", i, h_device_output[i]);
            }
            fclose(fp);
        }
    }

    free(h_device_input);
    free(h_device_output);
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void clearArray(unsigned char* arr, const unsigned int length) {
    unsigned int offset = 0;
    while (offset < length) {
        arr[offset] = 0;
        offset += 1;
    }
}

int main(int argc, char *argv[]) {
    const unsigned int length = 100000000; // 增加100倍
    unsigned char *arr_host = (unsigned char*)malloc(length * sizeof(unsigned char));
    srand(42);
    for(unsigned int i = 0; i < length; i++) {
        arr_host[i] = rand() % 256;
    }

    clock_t start = clock();

    clearArray(arr_host, length);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_199_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (unsigned int i = 0; i < 1000; i++) {
                fprintf(fp, "%u ", arr_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(arr_host);
    return 0;
}
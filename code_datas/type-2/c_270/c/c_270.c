#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Histogram(int* hist, unsigned char* img_in, int img_size) {
    for (int i = 0; i < img_size; ++i) {
        hist[img_in[i]]++;
    }
}

int main(int argc, char** argv) {
    int img_size = 10000000; // 增大图像大小以增加计算量
    unsigned char* h_img_in = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    int* h_hist = (int*)calloc(256, sizeof(int));

    srand(0);
    for (int i = 0; i < img_size; ++i) {
        h_img_in[i] = rand() % 256;
    }

    clock_t start = clock();

    Histogram(h_hist, h_img_in, img_size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_270_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_img_in[i]);
            }
            fprintf(fp, "\nHistogram:\n");
            for (int i = 0; i < 256; ++i) {
                fprintf(fp, "Value %d: %d\n", i, h_hist[i]);
            }
            fclose(fp);
        }
    }

    free(h_img_in);
    free(h_hist);

    return 0;
}
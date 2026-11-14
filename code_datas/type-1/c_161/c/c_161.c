#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void rgb2yuv_kernel(int img_size, unsigned char* gpu_img_in_r, unsigned char* gpu_img_in_g, unsigned char* gpu_img_in_b,
                    unsigned char* gpu_img_out_y, unsigned char* gpu_img_out_u, unsigned char* gpu_img_out_v) {
    unsigned char r, g, b;

    for (int index = 0; index < img_size; index++) {
        r = gpu_img_in_r[index];
        g = gpu_img_in_g[index];
        b = gpu_img_in_b[index];

        gpu_img_out_y[index] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        gpu_img_out_u[index] = (unsigned char)(-0.169 * r - 0.331 * g + 0.499 * b + 128);
        gpu_img_out_v[index] = (unsigned char)(0.499 * r - 0.418 * g - 0.0813 * b + 128);
    }
}

int main(int argc, char *argv[]) {
    int img_size = 10000000;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    unsigned char* gpu_img_in_r = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    unsigned char* gpu_img_in_g = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    unsigned char* gpu_img_in_b = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    unsigned char* gpu_img_out_y = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    unsigned char* gpu_img_out_u = (unsigned char*)malloc(img_size * sizeof(unsigned char));
    unsigned char* gpu_img_out_v = (unsigned char*)malloc(img_size * sizeof(unsigned char));

    srand(42);
    for (int i = 0; i < img_size; i++) {
        gpu_img_in_r[i] = rand() % 256;
        gpu_img_in_g[i] = rand() % 256;
        gpu_img_in_b[i] = rand() % 256;
    }

    clock_t start_time = clock();

    rgb2yuv_kernel(img_size, gpu_img_in_r, gpu_img_in_g, gpu_img_in_b, gpu_img_out_y, gpu_img_out_u, gpu_img_out_v);

    clock_t end_time = clock();

    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        FILE *fp = fopen("c_161_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%u ", gpu_img_out_y[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%u ", gpu_img_out_u[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%u ", gpu_img_out_v[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(gpu_img_in_r);
    free(gpu_img_in_g);
    free(gpu_img_in_b);
    free(gpu_img_out_y);
    free(gpu_img_out_u);
    free(gpu_img_out_v);
    return 0;
}
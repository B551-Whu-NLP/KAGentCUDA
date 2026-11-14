#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void yuv2rgb_kernel(int img_size, unsigned char* gpu_img_in_y, unsigned char* gpu_img_in_u, unsigned char* gpu_img_in_v,
    unsigned char* gpu_img_out_r, unsigned char* gpu_img_out_g, unsigned char* gpu_img_out_b, int verbose) {
    int rt, gt, bt;
    int rt2, gt2, bt2;
    clock_t start = clock();

    for (int index = 0; index < img_size; index++) {
        rt = (int)(gpu_img_in_y[index] + 1.402 * (gpu_img_in_v[index] - 128));
        gt = (int)(gpu_img_in_y[index] - 0.344 * (gpu_img_in_u[index] - 128) - 0.714 * (gpu_img_in_v[index] - 128));
        bt = (int)gpu_img_in_y[index] + 1.772 * (gpu_img_in_u[index] - 128);

        rt2 = (rt > 255) ? 255 : rt;
        gt2 = (gt > 255) ? 255 : gt;
        bt2 = (bt > 255) ? 255 : bt;

        gpu_img_out_r[index] = (rt2 < 0) ? 0 : rt2;
        gpu_img_out_g[index] = (gt2 < 0) ? 0 : gt2;
        gpu_img_out_b[index] = (bt2 < 0) ? 0 : bt2;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_173_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "(%d, %d, %d) -> (%d, %d, %d)\n", gpu_img_in_y[i], gpu_img_in_u[i], gpu_img_in_v[i],
                        gpu_img_out_r[i], gpu_img_out_g[i], gpu_img_out_b[i]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char *argv[]) {
    const int img_size = 10000000; // Increased from 10000 to 10000000
    unsigned char *gpu_img_in_y = (unsigned char*)malloc(img_size * sizeof(char));
    unsigned char *gpu_img_in_u = (unsigned char*)malloc(img_size * sizeof(char));
    unsigned char *gpu_img_in_v = (unsigned char*)malloc(img_size * sizeof(char));
    unsigned char *gpu_img_out_r = (unsigned char*)malloc(img_size * sizeof(char));
    unsigned char *gpu_img_out_g = (unsigned char*)malloc(img_size * sizeof(char));
    unsigned char *gpu_img_out_b = (unsigned char*)malloc(img_size * sizeof(char));

    for (int i = 0; i < img_size; ++i) {
        gpu_img_in_y[i] = 100;
        gpu_img_in_u[i] = 50;
        gpu_img_in_v[i] = 150;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    yuv2rgb_kernel(img_size, gpu_img_in_y, gpu_img_in_u, gpu_img_in_v, gpu_img_out_r, gpu_img_out_g, gpu_img_out_b, verbose);

    free(gpu_img_in_y);
    free(gpu_img_in_u);
    free(gpu_img_in_v);
    free(gpu_img_out_r);
    free(gpu_img_out_g);
    free(gpu_img_out_b);

    return 0;
}
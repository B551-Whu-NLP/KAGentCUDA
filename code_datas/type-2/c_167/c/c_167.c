#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void im2col(const int n, const float *data_im, const int height, const int width, const int ksize,
            const int pad, const int stride, const int height_col, const int width_col,
            float *data_col)
{
    clock_t start = clock();

    for (int index = 0; index < n; ++index)
    {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        float *data_col_ptr = data_col + (channel_out * height_col + h_out) * width_col + w_out;
        const float *data_im_ptr = data_im + (channel_in * height + h_in) * width + w_in;

        for (int i = 0; i < ksize; ++i)
        {
            for (int j = 0; j < ksize; ++j)
            {
                int h = h_in + i;
                int w = w_in + j;

                if (h >= 0 && w >= 0 && h < height && w < width)
                {
                    *(data_col_ptr + i * ksize + j) = *(data_im_ptr + i * width + j);
                }
                else
                {
                    *(data_col_ptr + i * ksize + j) = 0;
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char **argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    const int channels = 10;
    const int height = 2000;
    const int width = 2000;
    const int ksize = 5;
    const int pad = 2;
    const int stride = 1;
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int n = height_col * width_col * channels;

    float *data_im = (float *)malloc((long)channels * height * width * sizeof(float)); // ← 乘channels
    float *data_col = (float *)malloc((long)n * ksize * ksize * sizeof(float));

    srand(42);
    for (int i = 0; i < channels * height * width; ++i)
    { // ← 乘channels
        data_im[i] = (float)(rand() % 100);
    }

    im2col(n, data_im, height, width, ksize, pad, stride, height_col, width_col, data_col);

    if (verbose)
    {
        char output_file[256];
        strcpy(output_file, "c_167_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL)
        {
            int print_n = 100; // 只输出前100个
            for (int i = 0; i < print_n * ksize * ksize; ++i)
            {
                fprintf(fp, "%.2f ", data_col[i]);
                if ((i + 1) % (ksize * ksize) == 0)
                {
                    fprintf(fp, "\n");
                }
            }
            fclose(fp);
        }
    }

    free(data_im);
    free(data_col);

    return 0;
}
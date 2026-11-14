#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void convLayer_forward(int N, int M, int C, int H, int W, int K, float* X, float* Wk, float* Y, int verbose) {
    clock_t start = clock();

    int n, m, c, h, w, p, q;
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int iter = 0; iter < 1000; iter++) // 增加外层循环以延长执行时间
        for (n = 0; n < N; n++)
            for (m = 0; m < M; m++)
                for (h = 0; h < H_out; h++)
                    for (w = 0; w < W_out; w++) {
                        Y[n * M * H_out * W_out + m * H_out * W_out + h * W_out + w] = 0;

                        for (c = 0; c < C; c++)
                            for (p = 0; p < K; p++)
                                for (q = 0; q < K; q++)
                                    Y[n * M * H_out * W_out + m * H_out * W_out + h * W_out + w] +=
                                    X[n * C * H * W + c * H * W + (h + p) * W + (w + q)] * Wk[m * C * K * K + c * K * K + p * K + q];
                    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "%s_output.txt", "c_158");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < N * M * (H - K + 1) * (W - K + 1); i++) {
                fprintf(fp, "%.2f ", Y[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
}

int main(int argc, char** argv) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int N = 10;   // 增大尺寸
    int M = 15;
    int C = 20;
    int H = 50;
    int W = 50;
    int K = 5;

    float* X = (float*)malloc(N * C * H * W * sizeof(float));
    float* Wk = (float*)malloc(M * C * K * K * sizeof(float));
    float* Y = (float*)malloc(N * M * (H - K + 1) * (W - K + 1) * sizeof(float));

    for (int i = 0; i < N * C * H * W; i++) {
        X[i] = (float)(i + 1);
    }
    for (int i = 0; i < M * C * K * K; i++) {
        Wk[i] = (float)(i + 2);
    }

    convLayer_forward(N, M, C, H, W, K, X, Wk, Y, verbose);
    
    free(X);
    free(Wk);
    free(Y);
    return 0;
}
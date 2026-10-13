#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void dual_ascent(float* xn, float* xbar, float* y1, float* y2, float* img, float tau, float lambda, float theta, int w, int h, int nc) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                int i;
                float d1, d2, val;
                for (int z = 0; z < nc; z++) {
                    i = x + w * y + w * h * z;
                    d1 = (x + 1 < w ? y1[i] : 0.f) - (x > 0 ? y1[(x - 1) + w * y + w * h * z] : 0.f);
                    d2 = (y + 1 < h ? y2[i] : 0.f) - (y > 0 ? y2[x + w * (y - 1) + w * h * z] : 0.f);
                    val = xn[i];
                    xn[i] = ((val + tau * (d1 + d2)) + tau * lambda * img[i]) / (1.f + tau * lambda);
                    xbar[i] = xn[i] + theta * (xn[i] - val);
                }
            }
        }
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char* argv[]) {
    const int w = 256;
    const int h = 256;
    const int nc = 3;
    const float tau = 0.1f;
    const float lambda = 0.5f;
    const float theta = 0.2f;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* xn, * xbar, * y1, * y2, * img;
    xn = (float*)malloc(w * h * nc * sizeof(float));
    xbar = (float*)malloc(w * h * nc * sizeof(float));
    y1 = (float*)malloc(w * h * nc * sizeof(float));
    y2 = (float*)malloc(w * h * nc * sizeof(float));
    img = (float*)malloc(w * h * nc * sizeof(float));

    for (int i = 0; i < w * h * nc; ++i) {
        xn[i] = (float)(i % 100);
        xbar[i] = (float)(i % 100);
        y1[i] = (float)(i % 100);
        y2[i] = (float)(i % 100);
        img[i] = (float)(i % 100);
    }

    dual_ascent(xn, xbar, y1, y2, img, tau, lambda, theta, w, h, nc);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_180_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int k = 0; k < w * h * nc; k += 1) {
                fprintf(fp, "%f\n", xn[k]);
            }
            fclose(fp);
        }
    } 

    free(xn);
    free(xbar);
    free(y1);
    free(y2);
    free(img);
    return 0;
}
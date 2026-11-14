#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void fractal(const int width, const int frames, unsigned char* const pic) {
    clock_t start = clock();

    for (int i = 0; i < width * width * frames; i++) {
        const float Delta = 0.00304f;
        const float xMid = -0.055846456f;
        const float yMid = -0.668311119f;
        const int frame = i / (width * width);
        float delta = Delta * powf(0.975f, frame);
        const int col = i % width;
        const float xMin = xMid - delta;
        const float yMin = yMid - delta;
        const float dw = 2.0f * delta / width;
        const int row = (i / width) % width;
        const float cy = yMin + row * dw;
        const float cx = xMin + col * dw;
        float x = cx;
        float y = cy;
        float x2, y2;
        int count = 256;

        do {
            x2 = x * x;
            y2 = y * y;
            y = 2.0 * x * y + cy;
            x = x2 - y2 + cx;
            count--;
        } while ((count > 0) && ((x2 + y2) <= 5.0));

        pic[frame * width * width + row * width + col] = (unsigned char)count;
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    const int width = 512;
    const int frames = 20;
    unsigned char *pic = (unsigned char*)malloc(width * width * frames * sizeof(unsigned char));
    
    if (pic == NULL) {
        return 1;
    }

    srand(42);
    for (int i = 0; i < width * width * frames; i++) {
        pic[i] = rand() % 256;
    }

    fractal(width, frames, pic);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_169_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int frame = 0; frame < frames; frame++) {
                for (int row = 0; row < width; row++) {
                    for (int col = 0; col < width; col++) {
                        fprintf(fp, "%4d ", pic[frame * width * width + row * width + col]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
    free(pic);
    return 0;
}
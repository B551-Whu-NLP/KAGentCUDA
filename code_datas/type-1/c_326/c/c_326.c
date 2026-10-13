#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void waterElevationToDepth(const int nx_, const int ny_, float *h_ptr_, int h_pitch_, float *Bm_ptr_, int Bm_pitch_) {
    clock_t start = clock();
    
    for (int ti = 0; ti < nx_; ti++) {
        for (int tj = 0; tj < ny_; tj++) {
            float *h_row = h_ptr_ + h_pitch_ * tj;
            float *Bm_row = Bm_ptr_ + Bm_pitch_ * tj;
            h_row[ti] -= Bm_row[ti];
        }
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int nx = 1000;
    int ny = 1000;
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float *h_ptr = (float *)malloc(nx * ny * sizeof(float));
    float *Bm_ptr = (float *)malloc(nx * ny * sizeof(float));

    for (int i = 0; i < nx * ny; i++) {
        h_ptr[i] = 2 * i;
        Bm_ptr[i] = i;
    }

    waterElevationToDepth(nx, ny, h_ptr, nx, Bm_ptr, nx);

    if (verbose) {
        FILE *fp = fopen("c_326_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    fprintf(fp, "%8.2f\t", h_ptr[nx * i + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_ptr);
    free(Bm_ptr);

    return 0;
}
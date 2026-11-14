#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void bit8Channels(unsigned char* out, unsigned char* in, int channel, int n, int channels, int verbose) {
    clock_t start = clock();
    
    for (int i = 0; i < n; ++i) {
        int firstIndexToGrab = i * 8;
        unsigned char bit0 = (in[firstIndexToGrab + 0] & 0x01) << 0;
        unsigned char bit1 = (in[firstIndexToGrab + 1] & 0x01) << 1;
        unsigned char bit2 = (in[firstIndexToGrab + 2] & 0x01) << 2;
        unsigned char bit3 = (in[firstIndexToGrab + 3] & 0x01) << 3;
        unsigned char bit4 = (in[firstIndexToGrab + 4] & 0x01) << 4;
        unsigned char bit5 = (in[firstIndexToGrab + 5] & 0x01) << 5;
        unsigned char bit6 = (in[firstIndexToGrab + 6] & 0x01) << 6;
        unsigned char bit7 = (in[firstIndexToGrab + 7] & 0x01) << 7;
        unsigned char output = bit7 | bit6 | bit5 | bit4 | bit3 | bit2 | bit1 | bit0;
        
        int outputIndex = i * channels + channel - 1;
        out[outputIndex] = output;
    }
    
    clock_t end = clock();
    double time_taken = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    
    printf("Time: %.2f(ms)\n", time_taken);
    
    if (verbose) {
        char filename[256];
        sprintf(filename, "c_170_output.txt");
        FILE* fp = fopen(filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%02X ", out[i * channels + channel - 1]);
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    const int n = 1000000;
    const int channels = 3;
    unsigned char* in = (unsigned char*)malloc(n * 8 * sizeof(unsigned char));
    unsigned char* out = (unsigned char*)malloc(n * channels * sizeof(unsigned char));
    unsigned char values[] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                               0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
                               0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                               0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
                               0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                               0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                               0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
                               0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                               0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
                               0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77 };

    for (int i = 0; i < n * 8; ++i) {
        in[i] = values[i % 40]; 
    }


    free(in);
    free(out);

    return 0;
}
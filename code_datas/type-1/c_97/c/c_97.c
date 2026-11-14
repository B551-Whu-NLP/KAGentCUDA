#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void kernelXor(unsigned int key, char* input_str_cuda, unsigned char* possible_plaintext_str_cuda, int input_length) {
    int id;
    char *keyCharPtr;

    for (id = 0; id < input_length; id++) {
        int keyIndex = id % 4;
        keyCharPtr = ((char *)&key);
        char keyChar = keyCharPtr[keyIndex];
        possible_plaintext_str_cuda[id] = keyChar ^ input_str_cuda[id];
    }
}

int main(int argc, char *argv[]) {
    int input_length = 10000000;
    unsigned int key = 0x12345678;
    char* h_input_str = (char*)malloc(input_length * sizeof(char));
    unsigned char* h_possible_plaintext_str = (unsigned char*)malloc(input_length * sizeof(unsigned char));
    
    for (int i = 0; i < input_length; i++) {
        h_input_str[i] = i % 256;
    }

    clock_t start_time = clock();

    kernelXor(key, h_input_str, h_possible_plaintext_str, input_length);

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) * 1000 / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_97_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%02x ", h_possible_plaintext_str[i]);
            }
            fclose(fp);
        }
    }

    free(h_input_str);
    free(h_possible_plaintext_str);

    printf("Time: %.2f(ms)\n", time_taken);

    return 0;
}
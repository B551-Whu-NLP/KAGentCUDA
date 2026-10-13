
#include <stdio.h>
#include <stdlib.h>
#include "../headers/rodinia_backprop.h"

extern layer_size;

void load(BPNN *net)
{
    for (int i = 1; i <= net->input_n; i++) {
        net->input_units[i] = (float)drand48();
    }
    net->target[1] = (float)drand48();
}
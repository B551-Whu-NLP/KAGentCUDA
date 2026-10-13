// rodinia_backprop_facetrain.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../headers/rodinia_backprop.h"
#include "omp.h"

int layer_size = 0;

void backprop_face()
{
  BPNN *net = bpnn_create(layer_size, 16, 1);
  load(net);

  float out_err, hid_err;
  bpnn_train_kernel(net, &out_err, &hid_err);

  // 获取预测结果（输出层唯一神经元的激活值）
  float prediction = net->output_units[1];
  float target = net->target[1];

  // 保存结果：size, prediction, target, out_err, hid_err
  FILE *f = fopen("backprop_out.txt", "w");
  if (f)
  {
    fprintf(f, "%d\n", layer_size);
    fprintf(f, "%.10f\n", prediction);
    fprintf(f, "%.10f\n", target);
    fprintf(f, "%.10f\n", out_err);
    fprintf(f, "%.10f\n", hid_err);
    fclose(f);
  }

  bpnn_free(net);
}

int setup(int argc, char *argv[])
{
  int verbose = 0;
  int arg_start = 1;

  if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
  {
    verbose = 1;
    arg_start = 2; // size 参数从 argv[2] 开始
  }

  if (argc < arg_start + 1)
  {
    fprintf(stderr, "usage: %s <size>\n", argv[0]);
    exit(1);
  }

  layer_size = atoi(argv[1]);
  srand48(7); // 固定随机种子，确保 CPU/CUDA 输入一致
  if (verbose)
  {
    backprop_face();
  }
  return 0;
}
#pragma once

#include <common.h>

template<int GROUP_SIZE, int HIDDEN_DIM>
void run_simple_quantize_mxfp4(
  int num_rows,
  cutlass::bfloat16_t *input,
  uint8_t *output_data,
  cutlass::float_ue8m0_t *output_scale
);
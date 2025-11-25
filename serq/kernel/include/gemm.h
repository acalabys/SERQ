#pragma once

#include <common.h>

void matmul_host(
    int m, int n, int k,
    float alpha, float beta,
    const cutlass::float_e2m1_t* A_ptr, const cutlass::float_e2m1_t* B_ptr, const cutlass::bfloat16_t* C_ptr, const cutlass::bfloat16_t* D_ptr,
    const cutlass::float_ue8m0_t* SFA_ptr, const cutlass::float_ue8m0_t* SFB_ptr);
#pragma once

#include <common.h>

void matmul_host(
    uint32_t m, uint32_t n, uint32_t k,
    float alpha, float beta,
    const Fp4Storage* A_ptr, const Fp4Storage* B_ptr, const bfloat16* C_ptr, const bfloat16* D_ptr,
    const uint8_t* SFA_ptr, const uint8_t* SFB_ptr);
#include <torch/extension.h>

// Include all files
#include <gemm.h>

constexpr int kElementsPerVector = 2;

torch::Tensor matmul_host(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& SFA,
    const torch::Tensor& SFB
) {
    torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}, {C, "C", 2}, {SFA, "SFA", 3}, {SFB, "SFB", 4}});
    torch::checkDeviceType("matmul", {A, B, C, SFA, SFB}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}, {C, "C", 2}, {SFA, "SFA", 3}, {SFB, "SFB", 4}});
    uint32_t m = A.size(0);
    uint32_t n = B.size(0);
    uint32_t k = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
    auto D = torch::empty({m, n}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host(
        m, n, k, 
        1.0f, 0.0f, 
        A.data_ptr<Fp4Storage>(), 
        B.data_ptr<Fp4Storage>(), 
        C.data_ptr<bfloat16>(), 
        D.data_ptr<bfloat16>(), 
        SFA.data_ptr<uint8_t>(), 
        SFB.data_ptr<uint8_t>()
    );

    return D;
}
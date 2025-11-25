#include <torch/extension.h>

// Include all files
#include <gemm.h>
#include <mx_quant.h>

constexpr int kElementsPerVector = 2;

torch::Tensor matmul_lowrank(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& SFA,
    const torch::Tensor& SFB
) {
    torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}, {C, "C", 2}, {SFA, "SFA", 3}, {SFB, "SFB", 4}});
    torch::checkDeviceType("matmul", {A, B, C, SFA, SFB}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}, {C, "C", 2}, {SFA, "SFA", 3}, {SFB, "SFB", 4}});
    int m = A.size(0);
    int n = B.size(0);
    int k = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
    auto D = torch::empty({m, n}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host(
        m, n, k, 
        1.0f, 1.0f, 
        (cutlass::float_e2m1_t*)A.data_ptr<Fp4Storage>(), 
        (cutlass::float_e2m1_t*)B.data_ptr<Fp4Storage>(), 
        (cutlass::bfloat16_t*)C.data_ptr<at::BFloat16>(), 
        (cutlass::bfloat16_t*)D.data_ptr<at::BFloat16>(), 
        (cutlass::float_ue8m0_t*)SFA.data_ptr<uint8_t>(), 
        (cutlass::float_ue8m0_t*)SFB.data_ptr<uint8_t>()
    );

    return D;
}


torch::Tensor matmul(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& SFA,
    const torch::Tensor& SFB
) {
    torch::checkAllContiguous("matmul", {{A, "A", 0}, {B, "B", 1}, {SFA, "SFA", 2}, {SFB, "SFB", 3}});
    torch::checkDeviceType("matmul", {A, B, SFA, SFB}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("matmul", {{A, "A", 0}, {B, "B", 1}, {SFA, "SFA", 2}, {SFB, "SFB", 3}});
    int m = A.size(0);
    int n = B.size(0);
    int k = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
    auto D = torch::empty({m, n}, torch::dtype(torch::kBFloat16).device(A.device()));

    matmul_host(
        m, n, k, 
        1.0f, 0.0f, 
        (cutlass::float_e2m1_t*)A.data_ptr<Fp4Storage>(), 
        (cutlass::float_e2m1_t*)B.data_ptr<Fp4Storage>(),
        (cutlass::bfloat16_t*)D.data_ptr<at::BFloat16>(),
        (cutlass::bfloat16_t*)D.data_ptr<at::BFloat16>(),
        (cutlass::float_ue8m0_t*)SFA.data_ptr<uint8_t>(), 
        (cutlass::float_ue8m0_t*)SFB.data_ptr<uint8_t>()
    );

    return D;
}

std::tuple<torch::Tensor, torch::Tensor> simple_quantize_mxfp4(
    const torch::Tensor& input
) {
    torch::checkAllContiguous("simple_quantize_mxfp4_host", {{input, "input", 0}});
    torch::checkDeviceType("simple_quantize_mxfp4_host", {input}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("simple_quantize_mxfp4_host", {{input, "input", 0}});

    int num_rows = input.size(0);
    int hidden_dim = input.size(1);

    auto output = torch::empty({num_rows, hidden_dim / 2}, torch::dtype(torch::kUInt8).device(input.device()));
    auto scale = torch::empty({num_rows * hidden_dim / 32}, torch::dtype(torch::kUInt8).device(input.device()));

    if (hidden_dim == 2048) {
        run_simple_quantize_mxfp4<32, 2048>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 4096) {
        run_simple_quantize_mxfp4<32, 4096>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 5120) {
        run_simple_quantize_mxfp4<32, 5120>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 8192) {
        run_simple_quantize_mxfp4<32, 8192>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 11008) {
        run_simple_quantize_mxfp4<32, 11008>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 13824) {
        run_simple_quantize_mxfp4<32, 13824>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 14336) {
        run_simple_quantize_mxfp4<32, 14336>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else if (hidden_dim == 128) {
        run_simple_quantize_mxfp4<32, 128>(
            num_rows, 
            (cutlass::bfloat16_t*)input.data_ptr<at::BFloat16>(), 
            (uint8_t*)output.data_ptr<uint8_t>(), 
            reinterpret_cast<cutlass::float_ue8m0_t *>(scale.data_ptr<uint8_t>()));
    }
    else {
        throw std::runtime_error("Unsupported hidden dimension: " + std::to_string(hidden_dim));
    }
    return std::make_tuple(output, scale);
}


//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("matmul_lowrank", &matmul_lowrank,
          "input: (A: torch.Tensor(M x K, FP4_E2M1, CUDA), B: torch.Tensor(N x K, "
          "FP4_E2M1, CUDA), C: torch.Tensor(M x N, BFLOAT16, CUDA), SFA: torch.Tensor(M x 1, FP16, CUDA), SFB: torch.Tensor(N x 1, FP16, CUDA))\n"
          "output: torch.Tensor(M x N, BFLOAT16, CUDA)\n"
          "output = int4Unpacking(A) @ int4Unpacking(B)^T",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("SFA"), py::arg("SFB"));

    m.def("matmul", &matmul,
          "input: (A: torch.Tensor(M x K, FP4_E2M1, CUDA), B: torch.Tensor(N x K, "
          "FP4_E2M1, CUDA), SFA: torch.Tensor(M x 1, FP16, CUDA), SFB: torch.Tensor(N x 1, FP16, CUDA))\n"
          "output: torch.Tensor(M x N, BFLOAT16, CUDA)\n"
          "output = int4Unpacking(A) @ int4Unpacking(B)^T",
          py::arg("A"), py::arg("B"), py::arg("SFA"), py::arg("SFB"));

    m.def("simple_quantize_mxfp4", &simple_quantize_mxfp4,
          "input: torch.Tensor(M x N, BFLOAT16, CUDA)\n"
          "output: (quantized_data: torch.Tensor(M x ceil(N / 2), UINT8, CUDA), scale: "
          "torch.Tensor((M / 128 + 1) * 128 * N / 32, UINT8, CUDA))\n",
          py::arg("input"));

}
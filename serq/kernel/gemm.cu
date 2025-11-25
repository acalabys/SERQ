
#include <iostream>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/tensor_ref.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>

#include <gemm.h>

// Keep the same GEMM kernel configurations from the example
using namespace cute;

// A matrix configuration
using         ElementA    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using         LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 32;

// B matrix configuration
using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using         LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 32;

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;
using         ElementC    = cutlass::bfloat16_t;
using         LayoutCTag  = cutlass::layout::RowMajor;
using         LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

// Kernel functional config
using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm120;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;
using ClusterShape        = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void matmul_host(
    int m, int n, int k,
    float alpha, float beta,
    const cutlass::float_e2m1_t* A_ptr, 
    const cutlass::float_e2m1_t* B_ptr, 
    const cutlass::bfloat16_t* C_ptr, 
    const cutlass::bfloat16_t* D_ptr,
    const cutlass::float_ue8m0_t* SFA_ptr, const cutlass::float_ue8m0_t* SFB_ptr) {

    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using StrideD   = typename Gemm::GemmKernel::StrideD;

    using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape((int)m, (int)n, (int)k, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape((int)m, (int)n, (int)k, 1));

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        { // Mainloop arguments
            A_ptr, stride_A,
            B_ptr, stride_B,
            SFA_ptr, layout_SFA,
            SFB_ptr, layout_SFB
        },
        { // Epilogue arguments
            {alpha, beta},
            C_ptr, stride_C,
            D_ptr, stride_D
        }
    };

    Gemm gemm;

    auto status = gemm.run(arguments);
    ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));

    // if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    //     throw std::runtime_error("GEMM kernel cannot be implemented for the given problem size.");
    // }

    // size_t workspace_size = Gemm::get_workspace_size(arguments);
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // if (gemm.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) {
    //     throw std::runtime_error("Failed to initialize GEMM kernel.");
    // }

    // if (gemm.run() != cutlass::Status::kSuccess) {
    //     throw std::runtime_error("Failed to run GEMM kernel.");
    // }
}


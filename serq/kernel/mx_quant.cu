#include <cmath>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/device/tensor_compare.h>
#include <cutlass/util/reference/host/gett.hpp>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_norm.h>
#include <cutlass/util/tensor_view_io.h>
#include <mx_quant.h>

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

#define FP4_MAX 6
#define FP6_MAX 28
#define FP8_MAX 448

// MX data types from cutlass
using mx_float4_t = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
typedef cutlass::float_e2m1_t fp4_t;
typedef cutlass::float_e3m2_t fp6_t;
typedef cutlass::float_e4m3_t fp8_t;
typedef cutlass::float_ue8m0_t sf_t;
typedef cutlass::bfloat16_t bf16_t;

namespace cg = cooperative_groups;
using namespace cute;

struct PackFp4 {
  uint8_t low : 4;
  uint8_t high : 4;
};

HOST_DEVICE float fpmax(float a, float b) { return (a) > (b) ? (a) : (b); }

HOST_DEVICE float fpmin(float a, float b) { return (a) < (b) ? (a) : (b); }

HOST_DEVICE float clamp(float x, float a, float b) { return fpmax(a, fpmin(b, x)); }

template <typename T> HOST_DEVICE T abs(T x) { return x < (T)0 ? -x : x; }


namespace normal{
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // A matrix configuration
  using         ElementA    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
  using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
  using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
  using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
  using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
  using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
  // Kernel functional config
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

  // Kernel Perf config
  using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
  using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,                      
      ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
      >::CollectiveOp;
  
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
      >::CollectiveOp;
  
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,                                                   // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  
  // Reference device GEMM implementation type
  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  
  //
  // Data members
  //
  
  /// Initialization

  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  inline LayoutSFA get_layoutSFA(int M, int K) {
    return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, 128, K, 1));
  }
  inline LayoutSFB get_layoutSFB(int N, int K) {
    return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(128, N, K, 1));
  }
}


template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void simple_quantize_mxfp4_kernel(
  bf16_t *input,
  uint8_t *output_data,
  auto output_scale
) {
  
  constexpr int elements_per_thread = GROUP_SIZE;

  cg::thread_block cta = cg::this_thread_block();
  int tid = threadIdx.x;

  __shared__ uint8_t smem_raw[HIDDEN_DIM * sizeof(bf16_t)];
  bf16_t *input_smem = reinterpret_cast<bf16_t*>(smem_raw);

  bf16_t input_frag[elements_per_thread];

  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;
  output_data = output_data + row_id * (HIDDEN_DIM / 2);

  constexpr int bytes_per_iter = bdx * 16;
  constexpr int iters = HIDDEN_DIM * sizeof(bf16_t) / bytes_per_iter;

  #pragma unroll
  for (int i = 0; i < iters; ++i) {
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = 
        *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  cta.sync();

  int group_offset = tid * GROUP_SIZE;
  #pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    input_frag[i] = input_smem[group_offset + i];
  }

  float maxv = 0.0f;
  #pragma unroll
  for (int i = 0; i < elements_per_thread; ++i) {
    maxv = fmaxf(maxv, fabsf(static_cast<float>(input_frag[i])));
  }

  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterScale;
  float scale = 1.0f;
  if (maxv == 0.0f) {
    scale = 0.5f;
  } else {
    scale = ldexpf(1.0f, static_cast<int>(ceilf(log2f(maxv / FP4_MAX))));
  }
  float r_scale = 1.0f / scale;

  auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
  auto logical_coord2 = make_coord(0, 0);
  output_scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterScale(scale);

  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterN;

  PackFp4* packed_output_frag = reinterpret_cast<PackFp4*>(input_frag);
  
  float lower_bound = -FP4_MAX;
  float upper_bound = FP4_MAX;

  #pragma unroll
  for (int i = 0; i < elements_per_thread; i += 2) {
    float res0_f = clamp(static_cast<float>(input_frag[i]) * r_scale, lower_bound, upper_bound);
    float res1_f = clamp(static_cast<float>(input_frag[i + 1]) * r_scale, lower_bound, upper_bound);

    uint8_t packed_low = converterN(res0_f).storage;
    uint8_t packed_high = converterN(res1_f).storage;

    packed_output_frag[i / 2].low = packed_low;
    packed_output_frag[i / 2].high = packed_high;
  }

  float4* input_frag_float4 = reinterpret_cast<float4*>(input_frag);
  float4* output_data_float4 = reinterpret_cast<float4*>(output_data);

  output_data_float4[tid] = input_frag_float4[0];
}


template<int GROUP_SIZE, int HIDDEN_DIM>
void run_simple_quantize_mxfp4(
  int num_rows,
  bf16_t *input,
  uint8_t *output_data,
  sf_t *output_scale
) {
  static_assert(HIDDEN_DIM % GROUP_SIZE == 0, "HIDDEN_DIM must be divisible by GROUP_SIZE.");

  constexpr int num_threads_per_block = HIDDEN_DIM / GROUP_SIZE;
  
  dim3 gridDim(num_rows);
  dim3 blockDim(num_threads_per_block);

  cute::Tensor sfa_tensor = cute::make_tensor(output_scale, filter_zeros(normal::get_layoutSFA(num_rows, HIDDEN_DIM)));

  simple_quantize_mxfp4_kernel<num_threads_per_block, GROUP_SIZE, HIDDEN_DIM><<<gridDim, blockDim>>>(
    input,
    output_data,
    sfa_tensor
  );
}


template void run_simple_quantize_mxfp4<32, 128>(
  int num_rows,
  bf16_t *input,
  uint8_t *output_data,
  sf_t *output_scale
);

template void run_simple_quantize_mxfp4<32, 2048>(
  int num_rows,
  bf16_t *input,
  uint8_t *output_data,
  sf_t *output_scale
);

template void run_simple_quantize_mxfp4<32, 4096>(
    int num_rows,
    bf16_t *input,
    uint8_t *output_data,
    sf_t *output_scale
  );

template void run_simple_quantize_mxfp4<32, 5120>(
  int num_rows,
  bf16_t *input,
  uint8_t *output_data,
  sf_t *output_scale
);

template void run_simple_quantize_mxfp4<32, 8192>(
  int num_rows,
  bf16_t *input,
  uint8_t *output_data,
  sf_t *output_scale
);

template void run_simple_quantize_mxfp4<32, 11008>(
    int num_rows,
    bf16_t *input,
    uint8_t *output_data,
    sf_t *output_scale
  );

template void run_simple_quantize_mxfp4<32, 13824>(
  int num_rows,
  bf16_t *input,
  uint8_t *output_data,
  sf_t *output_scale
);

  template void run_simple_quantize_mxfp4<32, 14336>(
    int num_rows,
    bf16_t *input,
    uint8_t *output_data,
    sf_t *output_scale
  );
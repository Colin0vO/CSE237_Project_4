#include <aie_api/aie.hpp>
#include <aie_kernels/aie_kernel_utils.h>

#define DTYPE int8

// Make sure the following tile and intrinsic sizes match the sizes in the
// data layout transformations described in basic_mm.py.
constexpr unsigned m = 64;
constexpr unsigned k = 64;
constexpr unsigned n = 64;
constexpr unsigned r = 2;
constexpr unsigned s = 8;
constexpr unsigned t = 8;

using MMUL = aie::mmul<r, s, t, DTYPE, DTYPE>;

extern "C" {

//
// Dense layer kernel: Y = ReLU(X @ W)
//
// A, B, and C must be tiled into tiles of size r*s, s*t, and r*t,
// respectively (in our design, the DMA performs this tiling).
// Python side zeroes C before calling this kernel; here we perform
// GEMM accumulation and then apply ReLU.
void dense(const DTYPE *__restrict A,
           const DTYPE *__restrict B,
           DTYPE *__restrict C) {

  // Vector of zeros for ReLU: max(C, 0)
  const auto zeros = aie::zeros<DTYPE, MMUL::size_C>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned row = 0; row < m / r; row += 2) {
    for (unsigned col = 0; col < n / t; col += 2) {

      // Pointers to two rows of A tiles and two columns of B tiles
      const DTYPE *__restrict A0_ptr =
          A + ((row + 0) * (k / s) + 0) * MMUL::size_A;
      const DTYPE *__restrict A1_ptr =
          A + ((row + 1) * (k / s) + 0) * MMUL::size_A;
      const DTYPE *__restrict B0_ptr =
          B + (0 * (n / t) + (col + 0)) * MMUL::size_B;
      const DTYPE *__restrict B1_ptr =
          B + (0 * (n / t) + (col + 1)) * MMUL::size_B;

      // Load existing C tiles (Python has zeroed them already)
      const aie::vector<DTYPE, MMUL::size_C> C00_in =
          aie::load_v<MMUL::size_C>(
              C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C);
      const aie::vector<DTYPE, MMUL::size_C> C01_in =
          aie::load_v<MMUL::size_C>(
              C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C);
      const aie::vector<DTYPE, MMUL::size_C> C10_in =
          aie::load_v<MMUL::size_C>(
              C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C);
      const aie::vector<DTYPE, MMUL::size_C> C11_in =
          aie::load_v<MMUL::size_C>(
              C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C);

      MMUL C00(C00_in);
      MMUL C01(C01_in);
      MMUL C10(C10_in);
      MMUL C11(C11_in);

      // Iterate over the k dimension (microtiles)
      for (unsigned i = 0; i < k / s;
           i += 1,
           A0_ptr += MMUL::size_A,
           A1_ptr += MMUL::size_A,
           B0_ptr += (n / t) * MMUL::size_B,
           B1_ptr += (n / t) * MMUL::size_B) {

        const aie::vector<DTYPE, MMUL::size_A> A0 =
            aie::load_v<MMUL::size_A>(A0_ptr);
        const aie::vector<DTYPE, MMUL::size_A> A1 =
            aie::load_v<MMUL::size_A>(A1_ptr);
        const aie::vector<DTYPE, MMUL::size_B> B0 =
            aie::load_v<MMUL::size_B>(B0_ptr);
        const aie::vector<DTYPE, MMUL::size_B> B1 =
            aie::load_v<MMUL::size_B>(B1_ptr);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      // Convert accumulators back to DTYPE
      auto C00_vec = C00.template to_vector<DTYPE>();
      auto C01_vec = C01.template to_vector<DTYPE>();
      auto C10_vec = C10.template to_vector<DTYPE>();
      auto C11_vec = C11.template to_vector<DTYPE>();

      // Apply ReLU: max(C, 0)
      auto C00_relu = aie::max(C00_vec, zeros);
      auto C01_relu = aie::max(C01_vec, zeros);
      auto C10_relu = aie::max(C10_vec, zeros);
      auto C11_relu = aie::max(C11_vec, zeros);

      // Store back RESULT = ReLU(X @ W)
      aie::store_v(
          C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C, C00_relu);
      aie::store_v(
          C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C, C01_relu);
      aie::store_v(
          C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C, C10_relu);
      aie::store_v(
          C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C, C11_relu);
    }
  }
}

}
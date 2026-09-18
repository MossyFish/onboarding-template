#pragma once

#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> cells_;

public:
  Grid(std::size_t rows, std::size_t cols);

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  double* data() { return cells_.data(); }
  const double* data() const { return cells_.data(); }

  double& operator()(std::size_t i, std::size_t j);
  double  operator()(std::size_t i, std::size_t j) const;
};

Grid::Grid(std::size_t rows, std::size_t cols)
  : rows_{rows}
  , cols_{cols}
  , cells_(rows * cols, 0.0)
{}

double& Grid::operator()(std::size_t i, std::size_t j) {
  return cells_[i*cols_ + j];
}

double Grid::operator()(std::size_t i, std::size_t j) const {
  return cells_[i*cols_ + j];
}

namespace detail {

// Computes TWO interior rows (row, row+1) per call, sharing the loads of the
// two middle rows between both outputs -- row's "down" neighbor is the same
// data as row+1's "center", and row's "center" is the same data as row+1's
// "up" neighbor. Regular (non-streaming) stores on purpose: this data gets
// read again as the *next* timestep's input, so it should stay cached.
__attribute__((target("avx2,fma")))
inline void stencil_row_pair_avx2(
  const double* __restrict old_data, double* __restrict new_data,
  int row, int cols_i
) {
  const int rowM1 = (row - 1) * cols_i;
  const int row0  = row * cols_i;
  const int row1  = (row + 1) * cols_i;
  const int rowP2 = (row + 2) * cols_i;

  new_data[row0] = old_data[row0];
  new_data[row0 + cols_i - 1] = old_data[row0 + cols_i - 1];
  new_data[row1] = old_data[row1];
  new_data[row1 + cols_i - 1] = old_data[row1 + cols_i - 1];

  const int end = cols_i - 1;
  int j = 1;

  // Peel: scalar until (row0 + j) is 32-byte aligned-equivalent (multiple of 4).
  while (j < end && (j % 4 != 0)) {
    new_data[row0 + j] = 0.5 * old_data[row0 + j] +
      0.125 * (old_data[rowM1 + j] + old_data[row1 + j] +
               old_data[row0 + j - 1] + old_data[row0 + j + 1]);
    new_data[row1 + j] = 0.5 * old_data[row1 + j] +
      0.125 * (old_data[row0 + j] + old_data[rowP2 + j] +
               old_data[row1 + j - 1] + old_data[row1 + j + 1]);
    ++j;
  }

  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d eighth = _mm256_set1_pd(0.125);

  for (; j + 4 <= end; j += 4) {
    __m256d top    = _mm256_loadu_pd(&old_data[rowM1 + j]);
    __m256d mid1   = _mm256_loadu_pd(&old_data[row0 + j]);
    __m256d mid1L  = _mm256_loadu_pd(&old_data[row0 + j - 1]);
    __m256d mid1R  = _mm256_loadu_pd(&old_data[row0 + j + 1]);
    __m256d mid2   = _mm256_loadu_pd(&old_data[row1 + j]);
    __m256d mid2L  = _mm256_loadu_pd(&old_data[row1 + j - 1]);
    __m256d mid2R  = _mm256_loadu_pd(&old_data[row1 + j + 1]);
    __m256d bottom = _mm256_loadu_pd(&old_data[rowP2 + j]);

    // row: center=mid1, up=top, down=mid2 (shared), left/right=mid1L/R
    __m256d sumA = _mm256_add_pd(_mm256_add_pd(top, mid2), _mm256_add_pd(mid1L, mid1R));
    __m256d outA = _mm256_fmadd_pd(eighth, sumA, _mm256_mul_pd(half, mid1));
    _mm256_storeu_pd(&new_data[row0 + j], outA);

    // row+1: center=mid2, up=mid1 (shared), down=bottom, left/right=mid2L/R
    __m256d sumB = _mm256_add_pd(_mm256_add_pd(mid1, bottom), _mm256_add_pd(mid2L, mid2R));
    __m256d outB = _mm256_fmadd_pd(eighth, sumB, _mm256_mul_pd(half, mid2));
    _mm256_storeu_pd(&new_data[row1 + j], outB);
  }

  // Tail: whatever's left.
  for (; j < end; ++j) {
    new_data[row0 + j] = 0.5 * old_data[row0 + j] +
      0.125 * (old_data[rowM1 + j] + old_data[row1 + j] +
               old_data[row0 + j - 1] + old_data[row0 + j + 1]);
    new_data[row1 + j] = 0.5 * old_data[row1 + j] +
      0.125 * (old_data[row0 + j] + old_data[rowP2 + j] +
               old_data[row1 + j - 1] + old_data[row1 + j + 1]);
  }
}

}  // namespace detail

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const double* __restrict old_data = old_grid.data();
  double* __restrict new_data = new_grid.data();

  const int rows_i = static_cast<int>(old_grid.rows());
  const int cols_i = static_cast<int>(old_grid.cols());

  std::memcpy(new_data, old_data, cols_i * sizeof(double));
  std::memcpy(new_data + (rows_i - 1) * cols_i, old_data + (rows_i - 1) * cols_i, cols_i * sizeof(double));

  const int interior_count = rows_i - 2;
  if (interior_count <= 0) {
    return;
  }

  const int paired_count = interior_count / 2;

  #pragma omp parallel for
  for (int p = 0; p < paired_count; ++p) {
    detail::stencil_row_pair_avx2(old_data, new_data, 1 + 2 * p, cols_i);
  }

  // Odd number of interior rows: the last one has no partner to pair with.
  if (interior_count % 2 == 1) {
    const int row = rows_i - 2;
    const int row_offset = row * cols_i;

    new_data[row_offset] = old_data[row_offset];
    new_data[row_offset + cols_i - 1] = old_data[row_offset + cols_i - 1];

    #pragma omp simd
    for (int j = 1; j < cols_i - 1; ++j) {
      const int index = row_offset + j;
      new_data[index] = 0.5 * old_data[index] +
        0.125 * (old_data[index - cols_i] + old_data[index + cols_i] +
                 old_data[index - 1] + old_data[index + 1]);
    }
  }
}

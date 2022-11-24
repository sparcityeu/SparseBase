#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/reorder/reorder_heatmap.h"
#include "sparsebase/reorder/reorderer.h"
#ifdef USE_CUDA
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#endif

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
TEST(ReorderHeatmap, Instance){
ReorderHeatmap<int, int, int, float> heatmapper(3);
int no_reoder_perm_arr[3] = {0,1,2};
format::Array<int> no_reorder_perm(3, no_reoder_perm_arr, format::kNotOwned);
auto heatmap = heatmapper.Get(&global_csr, &no_reorder_perm, &no_reorder_perm, {global_csr.get_context()}, true);
auto heatmap_arr = heatmap->As<format::Array>()->get_vals();
for (int i =0; i< 9 ; i++) EXPECT_EQ(heatmap_arr[i], heatmap_no_order_true[i]);
#ifdef USE_CUDA
// Try with a cuda array and COO
  context::CUDAContext g0{0};
  format::CUDAArray<int>* cuda_arr = no_reorder_perm.Convert<format::CUDAArray>(&g0);
  heatmap = heatmapper.Get(&global_coo, cuda_arr, cuda_arr, {global_csr.get_context()}, true);
  heatmap_arr = heatmap->As<format::Array>()->get_vals();
  for (int i =0; i< 9 ; i++) EXPECT_EQ(heatmap_arr[i], heatmap_no_order_true[i]);
#endif
}

TEST(ReorderHeatmap, InstanceTwoReorders){
ReorderHeatmap<int, int, int, float> heatmapper(3);
int no_reoder_perm_arr[3] = {0,1,2};
format::Array<int> r_reorder_perm(3, r_reorder_vector, format::kNotOwned);
format::Array<int> c_reorder_perm(3, c_reorder_vector, format::kNotOwned);
auto heatmap = heatmapper.Get(&global_csr, &r_reorder_perm, &c_reorder_perm, {global_csr.get_context()}, true);
auto heatmap_arr = heatmap->As<format::Array>()->get_vals();
for (int i =0; i< 9 ; i++) EXPECT_EQ(heatmap_arr[i], heatmap_rc_order_true[i]);
#ifdef USE_CUDA
// Try with a cuda array and COO
  context::CUDAContext g0{0};
  format::CUDAArray<int>* cuda_arr_r = r_reorder_perm.Convert<format::CUDAArray>(&g0);
  format::CUDAArray<int>* cuda_arr_c = c_reorder_perm.Convert<format::CUDAArray>(&g0);
  heatmap = heatmapper.Get(&global_coo, cuda_arr_r, cuda_arr_c, {global_csr.get_context()}, true);
  heatmap_arr = heatmap->As<format::Array>()->get_vals();
  for (int i =0; i< 9 ; i++) EXPECT_EQ(heatmap_arr[i], heatmap_rc_order_true[i]);
#endif
}

TEST(ReorderHeatmap, InstanceDefaultConstructor) {
  ReorderHeatmap<int, int, int, float> heatmapper;
  int no_reoder_perm_arr[3] = {0, 1, 2};
  format::Array<int> no_reorder_perm(3, no_reoder_perm_arr, format::kNotOwned);
  auto heatmap = heatmapper.Get(&global_csr, &no_reorder_perm, &no_reorder_perm,
                                {global_csr.get_context()}, true);
  auto heatmap_arr = heatmap->As<format::Array>()->get_vals();
  for (int i = 0; i < 9; i++)
    EXPECT_EQ(heatmap_arr[i], heatmap_no_order_true[i]);
#ifdef USE_CUDA
// Try with a cuda array and COO
  context::CUDAContext g0{0};
  format::CUDAArray<int>* cuda_arr = no_reorder_perm.Convert<format::CUDAArray>(&g0);
  heatmap = heatmapper.Get(&global_coo, cuda_arr, cuda_arr, {global_csr.get_context()}, true);
  heatmap_arr = heatmap->As<format::Array>()->get_vals();
  for (int i =0; i< 9 ; i++) EXPECT_EQ(heatmap_arr[i], heatmap_no_order_true[i]);
#endif
}

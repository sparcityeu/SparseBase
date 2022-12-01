#include <iostream>
#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/generic_reorder.h"
#include "sparsebase/reorder/gray_reorder.h"
#include "sparsebase/reorder/reorder_heatmap.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
#ifdef USE_CUDA
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#endif

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
TEST(ReorderBase, RCMReorder) {
  EXPECT_NO_THROW(EXECUTE_AND_DELETE(
      ReorderBase::Reorder<RCMReorder>({}, &global_csr, {&cpu_context}, true)));
  auto order =
      ReorderBase::Reorder<RCMReorder>({}, &global_csr, {&cpu_context}, true);
  check_reorder(order, n);
}

#ifdef USE_METIS
TEST(ReorderBase, MetisReorder) {
  if (typeid(metis::idx_t) == typeid(int)) {
    EXPECT_NO_THROW(EXECUTE_AND_DELETE(ReorderBase::Reorder<MetisReorder>(
        {}, &global_csr, {&cpu_context}, true)));
    auto order = ReorderBase::Reorder<MetisReorder>({}, &global_csr,
                                                    {&cpu_context}, true);
    check_reorder(order, n);
  } else {
    auto global_csr_64_bit =
        global_csr.Convert<sparsebase::format::CSR, int64_t, int64_t, int64_t>(
            false);
    auto order = ReorderBase::Reorder<MetisReorder>({}, global_csr_64_bit,
                                                    {&cpu_context}, true);
    check_reorder(order, (int64_t)n);
  }
}
#endif

#ifdef USE_RABBIT_ORDER
TEST(ReorderBase, RabbitReorder) {
  EXPECT_NO_THROW(EXECUTE_AND_DELETE(ReorderBase::Reorder<RabbitReorder>(
      {}, &global_csr, {&cpu_context}, true)));
  auto order = ReorderBase::Reorder<RabbitReorder>({}, &global_csr,
                                                   {&cpu_context}, true);
  check_reorder(order, n);
}
#endif

#ifdef USE_AMD_ORDER
TEST(ReorderBase, AMDReorder) {
  EXPECT_NO_THROW(EXECUTE_AND_DELETE(
      ReorderBase::Reorder<AMDReorder>({}, &global_csr, {&cpu_context}, true)));
  auto order =
      ReorderBase::Reorder<AMDReorder>({}, &global_csr, {&cpu_context}, true);
  check_reorder(order, n);
}
#endif

TEST(ReorderBase, DegreeReorder) {
  EXPECT_NO_THROW(EXECUTE_AND_DELETE(ReorderBase::Reorder<DegreeReorder>(
      {true}, &global_csr, {&cpu_context}, true)));
  auto order = ReorderBase::Reorder<DegreeReorder>({false}, &global_csr,
                                                   {&cpu_context}, true);
  check_reorder(order, n);
}

TEST(ReorderBase, GrayReorder) {
  EXPECT_NO_THROW(EXECUTE_AND_DELETE(ReorderBase::Reorder<GrayReorder>(
      {BitMapSize::BitSize16, 48, 32}, &global_csr, {&cpu_context}, true)));
  auto order = ReorderBase::Reorder<GrayReorder>(
      {BitMapSize::BitSize16, 48, 32}, &global_csr, {&cpu_context}, true);
  check_reorder(order, n);
}

TEST(ReorderBase, ReorderCached) {
  auto out_no_convert =
      ReorderBase::ReorderCached<RCMReorder>({}, &global_csr, {&cpu_context});
  EXPECT_EQ(out_no_convert.first.size(), 0);
  auto order =
      ReorderBase::ReorderCached<RCMReorder>({}, &global_coo, {&cpu_context});
  check_reorder(order.second, n);
  compare_csr(order.first[0]->Convert<format::CSR>(), &global_csr);
}

TEST(ReorderBase, Heatmap) {
  int no_reoder_perm_arr[3] = {0, 1, 2};
  format::Array<int> no_reorder_perm(3, no_reoder_perm_arr, format::kNotOwned);
  auto heatmap_arr = ReorderBase::Heatmap<float>(
                         &global_csr, &no_reorder_perm, &no_reorder_perm, 3,
                         {global_csr.get_context()}, true)
                         ->get_vals();
  for (int i = 0; i < 9; i++)
    EXPECT_EQ(heatmap_arr[i], heatmap_no_order_true[i]);
#ifdef USE_CUDA
  // Try with a cuda array and COO
  context::CUDAContext g0{0};
  format::CUDAArray<int>* cuda_arr =
      no_reorder_perm.Convert<format::CUDAArray>(&g0);
  heatmap_arr = ReorderBase::Heatmap<float>(&global_coo, cuda_arr, cuda_arr, 3,
                                            {global_csr.get_context()}, true)
                    ->get_vals();
  for (int i = 0; i < 9; i++)
    EXPECT_EQ(heatmap_arr[i], heatmap_no_order_true[i]);
#endif
}
TEST(ReorderBase, HeatmapTwoReorders) {
  ReorderHeatmap<int, int, int, float> heatmapper(3);
  int no_reoder_perm_arr[3] = {0, 1, 2};
  format::Array<int> r_reorder_perm(3, r_reorder_vector, format::kNotOwned);
  format::Array<int> c_reorder_perm(3, c_reorder_vector, format::kNotOwned);
  auto heatmap_arr =
      ReorderBase::Heatmap<float>(&global_csr, &r_reorder_perm, &c_reorder_perm,
                                  3, {global_csr.get_context()}, true)
          ->get_vals();
  for (int i = 0; i < 9; i++)
    EXPECT_EQ(heatmap_arr[i], heatmap_rc_order_true[i]);
#ifdef USE_CUDA
  // Try with a cuda array and COO
  context::CUDAContext g0{0};
  format::CUDAArray<int>* cuda_arr_r =
      r_reorder_perm.Convert<format::CUDAArray>(&g0);
  format::CUDAArray<int>* cuda_arr_c =
      c_reorder_perm.Convert<format::CUDAArray>(&g0);
  heatmap_arr = ReorderBase::Heatmap<float>(&global_coo, cuda_arr_r, cuda_arr_c,
                                            3, {global_csr.get_context()}, true)
                    ->get_vals();
  for (int i = 0; i < 9; i++)
    EXPECT_EQ(heatmap_arr[i], heatmap_rc_order_true[i]);
#endif
}

TEST(ReorderBase, Permute1D) {
  // no conversion of output
  EXPECT_NO_THROW(ReorderBase::Permute1D(inverse_perm_array, &orig_arr,
                                         {&cpu_context}, true));
  // check output of permutation
  format::Format* inv_arr_fp = ReorderBase::Permute1D(
      inverse_perm_array, &orig_arr, {&cpu_context}, true);

  auto* inv_arr = inv_arr_fp->AsAbsolute<format::Array<float>>();
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(inv_arr->get_vals()[i], reordered_array[i]);
  }
  // converting output to possible format
  EXPECT_NO_THROW(ReorderBase::Permute1D<format::Array>(
      inverse_perm_array, &orig_arr, {&cpu_context}, true));
  EXPECT_EQ((ReorderBase::Permute1D<format::Array>(
                 inverse_perm_array, &orig_arr, {&cpu_context}, true))
                ->get_id(),
            (format::Array<float>::get_id_static()));
  inv_arr = ReorderBase::Permute1D<format::Array>(inverse_perm_array, &orig_arr,
                                                  {&cpu_context}, true);

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(inv_arr->get_vals()[i], reordered_array[i]);
  }
  EXPECT_THROW(ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &global_csr,
                                                  {&cpu_context}, false, false),
               utils::TypeException);
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &global_csr,
                                                  {&cpu_context}, false, true),
               utils::ConversionException);
  EXPECT_THROW(ReorderBase::Permute1D<TestFormatOrderOne>(
                   inverse_perm_array, &orig_arr, {&cpu_context}, true, true),
               utils::ConversionException);
  EXPECT_THROW(ReorderBase::Permute1D<TestFormatOrderOne>(
                   inverse_perm_array, &orig_arr, {&cpu_context}, true, false),
               utils::TypeException);
  // passing a format that isn't convertable
  TestFormatOrderOne<int> f;
  EXPECT_THROW(
      ReorderBase::Permute1D(r_reorder_vector, &f, {&cpu_context}, true),
      utils::FunctionNotFoundException);
  EXPECT_THROW(
      ReorderBase::Permute1D(r_reorder_vector, &f, {&cpu_context}, false),
      utils::FunctionNotFoundException);
}
TEST(ReorderBase, Permute1DCached) {
  // no conversion of output
  auto output_no_convert_input = ReorderBase::Permute1DCached(
      inverse_perm_array, &orig_arr, {&cpu_context});
  EXPECT_EQ(output_no_convert_input.first.size(), 0);
// converting input to possible format
#ifdef USE_CUDA
  context::CUDAContext g0{0};
  auto cuda_arr = orig_arr.Convert<format::CUDAArray>(&g0);
  auto output_convert_input = ReorderBase::Permute1DCached(
      inverse_perm_array, cuda_arr, {&cpu_context});
  EXPECT_NE(output_convert_input.first.size(), 0);
  auto transformed_format =
      output_convert_input.second->Convert<format::Array>();
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(transformed_format->get_vals()[i], reordered_array[i]);
  }
  auto output_convert_input_output =
      ReorderBase::Permute1DCached<format::Array>(inverse_perm_array, cuda_arr,
                                                  {&cpu_context});
  auto transformed_format_input_output = output_convert_input_output.second;
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(transformed_format_input_output->get_vals()[i],
              reordered_array[i]);
  }
  EXPECT_NE(output_convert_input_output.first.size(), 0);
#endif
}
TEST(ReorderBase, Permute2D) {
  // no conversion of output
  EXPECT_NO_THROW(ReorderBase::Permute2D(r_reorder_vector, &global_csr,
                                         {&cpu_context}, true));
  EXPECT_NO_THROW(ReorderBase::Permute2D(r_reorder_vector, &global_csr,
                                         {&cpu_context}, false));
  // check output of permutation
  auto transformed_format =
      ReorderBase::Permute2D(r_reorder_vector, &global_csr, {&cpu_context},
                             true)
          ->Convert<format::CSR>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), r_reorder_vector, n);
  // converting output to possible format
  EXPECT_NO_THROW(ReorderBase::Permute2D<format::CSR>(
      r_reorder_vector, &global_csr, {&cpu_context}, false));
  EXPECT_NO_THROW(ReorderBase::Permute2D<format::CSR>(
      r_reorder_vector, &global_csr, {&cpu_context}, true));
  EXPECT_THROW(ReorderBase::Permute2D<format::COO>(
                   r_reorder_vector, &global_csr, {&cpu_context}, true, false),
               utils::TypeException);
  EXPECT_NO_THROW(ReorderBase::Permute2D<format::COO>(
      r_reorder_vector, &global_csr, {&cpu_context}, true, true));
  EXPECT_EQ((ReorderBase::Permute2D<format::CSR>(r_reorder_vector, &global_csr,
                                                 {&cpu_context}, true))
                ->get_id(),
            (format::CSR<int, int, int>::get_id_static()));
  transformed_format = ReorderBase::Permute2D<format::CSR>(
      r_reorder_vector, &global_csr, {&cpu_context}, true);
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), r_reorder_vector, n);
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &global_csr,
                                                  {&cpu_context}, true, true),
               utils::ConversionException);
  // passing a format that isn't convertable
  TestFormat<int, int, int> f;
  EXPECT_THROW(ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &f,
                                                  {&cpu_context}, true),
               utils::FunctionNotFoundException);
}
TEST(ReorderBase, Permute2DCached) {
  // no conversion of output
  auto output_no_convert_input = ReorderBase::Permute2DCached(
      r_reorder_vector, &global_csr, {&cpu_context});
  EXPECT_EQ(output_no_convert_input.first.size(), 0);
  // converting input to possible format
  auto output_convert_input = ReorderBase::Permute2DCached(
      r_reorder_vector, &global_coo, {&cpu_context});
  EXPECT_EQ(output_convert_input.first.size(), 1);
  auto transformed_format = output_convert_input.second->Convert<format::CSR>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), r_reorder_vector, n);
  // not converting output to possible format
  EXPECT_THROW(ReorderBase::Permute2DCached<format::COO>(
                   r_reorder_vector, &global_coo, {&cpu_context}, false),
               utils::TypeException);
  // converting output to possible format
  auto output_convert_input_output = ReorderBase::Permute2DCached<format::COO>(
      r_reorder_vector, &global_coo, {&cpu_context}, true);
  EXPECT_EQ(output_convert_input_output.first.size(), 1);
  auto transformed_format_input_output =
      output_convert_input.second->Convert<format::CSR>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format_input_output->get_row_ptr(),
      global_csr.get_col(), transformed_format_input_output->get_col(),
      r_reorder_vector, n);
  compare_csr(output_convert_input_output.first[0]->Convert<format::CSR>(),
              &global_csr);
}
TEST(ReorderBase, InversePermutation) {
  auto perm = ReorderBase::InversePermutation(inverse_perm_array,
                                              orig_arr.get_dimensions()[0]);
  for (int i = 0; i < array_length; i++) {
    EXPECT_EQ(perm[i], perm_array[i]);
  }
}
TEST(ReorderBase, Permute2DRowColWise) {
  // no conversion of output
  EXPECT_NO_THROW(ReorderBase::Permute2DRowColumnWise(
      r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true));
  // check output of permutation
  auto transformed_format =
      ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector,
                                          &global_csr, {&cpu_context}, true)
          ->Convert<format::CSR>();
  for (int i = 0; i < n + 1; i++) {
    EXPECT_EQ(transformed_format->get_row_ptr()[i], rc_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(transformed_format->get_col()[i], rc_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], rc_vals[i]);
  }
  // converting output to possible format
  EXPECT_NO_THROW(ReorderBase::Permute2DRowColumnWise(
      r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true));
  EXPECT_EQ(
      (ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector,
                                           &global_csr, {&cpu_context}, true))
          ->get_id(),
      (format::CSR<int, int, int>::get_id_static()));
  transformed_format =
      ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector,
                                          &global_csr, {&cpu_context}, true)
          ->Convert<format::CSR>();
  for (int i = 0; i < n + 1; i++) {
    EXPECT_EQ(transformed_format->get_row_ptr()[i], rc_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(transformed_format->get_col()[i], rc_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], rc_vals[i]);
  }
  // converting output to illegal format (boolean is off)
  EXPECT_THROW(ReorderBase::Permute2DRowColumnWise<TestFormat>(
                   r_reorder_vector, c_reorder_vector, &global_csr,
                   {&cpu_context}, true, false),
               utils::TypeException);
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(ReorderBase::Permute2DRowColumnWise<TestFormat>(
                   r_reorder_vector, c_reorder_vector, &global_csr,
                   {&cpu_context}, true, true),
               utils::ConversionException);
  // passing a format that isn't convertable
  TestFormat<int, int, int> f;
  EXPECT_THROW(
      ReorderBase::Permute2DRowColumnWise<TestFormat>(
          r_reorder_vector, c_reorder_vector, &f, {&cpu_context}, true, true),
      utils::FunctionNotFoundException);
  EXPECT_THROW(
      ReorderBase::Permute2DRowColumnWise<TestFormat>(
          r_reorder_vector, c_reorder_vector, &f, {&cpu_context}, false, true),
      utils::FunctionNotFoundException);
}
TEST(ReorderBase, Permute2DCachedRowColWise) {
  // no conversion of output
  auto output_no_convert_input = ReorderBase::Permute2DRowColumnWiseCached(
      r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context});
  EXPECT_EQ(output_no_convert_input.first.size(), 0);
  // converting input to possible format
  auto output_convert_input = ReorderBase::Permute2DRowColumnWiseCached(
      r_reorder_vector, c_reorder_vector, &global_coo, {&cpu_context});
  EXPECT_EQ(output_convert_input.first.size(), 1);
  auto transformed_format = output_convert_input.second->Convert<format::CSR>();
  // converting output to possible format
  auto output_convert_input_output =
      ReorderBase::Permute2DRowColumnWiseCached<format::CSR>(
          r_reorder_vector, c_reorder_vector, &global_coo, {&cpu_context});
  EXPECT_EQ(output_convert_input_output.first.size(), 1);
  auto transformed_format_input_output =
      output_convert_input.second->Convert<format::CSR>();
  compare_csr(output_convert_input_output.first[0]->Convert<format::CSR>(),
              &global_csr);
}
TEST(ReorderBase, Permute2DRowWise) {
  // no conversion of output
  EXPECT_NO_THROW(ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr,
                                                {&cpu_context}, true));
  // check output of permutation
  auto transformed_format =
      ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr,
                                    {&cpu_context}, true)
          ->Convert<format::CSR>();
  for (int i = 0; i < n + 1; i++) {
    EXPECT_EQ(transformed_format->get_row_ptr()[i], r_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(transformed_format->get_col()[i], r_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], r_vals[i]);
  }
  // converting output to possible format
  EXPECT_NO_THROW(ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr,
                                                {&cpu_context}, true));
  EXPECT_EQ((ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr,
                                           {&cpu_context}, true))
                ->get_id(),
            (format::CSR<int, int, int>::get_id_static()));
  transformed_format = ReorderBase::Permute2DRowWise(
                           r_reorder_vector, &global_csr, {&cpu_context}, true)
                           ->Convert<format::CSR>();
  for (int i = 0; i < n + 1; i++) {
    EXPECT_EQ(transformed_format->get_row_ptr()[i], r_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(transformed_format->get_col()[i], r_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], r_vals[i]);
  }
  // converting output to illegal format (boolean is off)
  EXPECT_THROW(ReorderBase::Permute2DRowWise<TestFormat>(
                   r_reorder_vector, &global_csr, {&cpu_context}, true, false),
               utils::TypeException);
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(ReorderBase::Permute2DRowWise<TestFormat>(
                   r_reorder_vector, &global_csr, {&cpu_context}, true, true),
               utils::ConversionException);
  // passing a format that isn't convertable
  TestFormat<int, int, int> f;
  EXPECT_THROW(ReorderBase::Permute2DRowWise<TestFormat>(
                   r_reorder_vector, &f, {&cpu_context}, true, true),
               utils::FunctionNotFoundException);
  EXPECT_THROW(ReorderBase::Permute2DRowWise<TestFormat>(
                   r_reorder_vector, &f, {&cpu_context}, true, false),
               utils::FunctionNotFoundException);
}
TEST(ReorderBase, Permute2DCachedRowWise) {
  // no conversion of output
  auto output_no_convert_input = ReorderBase::Permute2DRowWiseCached(
      r_reorder_vector, &global_csr, {&cpu_context});
  EXPECT_EQ(output_no_convert_input.first.size(), 0);
  // converting input to possible format
  auto output_convert_input = ReorderBase::Permute2DRowWiseCached(
      r_reorder_vector, &global_coo, {&cpu_context});
  EXPECT_EQ(output_convert_input.first.size(), 1);
  auto transformed_format = output_convert_input.second->Convert<format::CSR>();
  // converting output to possible format
  auto output_convert_input_output =
      ReorderBase::Permute2DRowWiseCached<format::CSR>(
          r_reorder_vector, &global_coo, {&cpu_context});
  EXPECT_EQ(output_convert_input_output.first.size(), 1);
  auto transformed_format_input_output =
      output_convert_input.second->Convert<format::CSR>();
  compare_csr(output_convert_input_output.first[0]->Convert<format::CSR>(),
              &global_csr);
}

// TEST(ReorderBase, Permute2DColWise) {
//  // no conversion of output
//  EXPECT_NO_THROW(ReorderBase::Permute2DColWise(c_reorder_vector,
//  &global_csr, {&cpu_context}));
//  // check output of permutation
//  auto transformed_format =
//  ReorderBase::Permute2DColWise(c_reorder_vector,
//  &global_csr, {&cpu_context})->Convert<format::CSR>(); for (int i = 0; i <
//  n+1; i++){
//    EXPECT_EQ(transformed_format->get_row_ptr()[i], c_row_ptr[i]);
//  }
//  for (int i = 0; i < nnz; i++){
//    EXPECT_EQ(transformed_format->get_col()[i], c_cols[i]);
//    EXPECT_EQ(transformed_format->get_vals()[i], c_vals[i]);
//  }
//  // converting output to possible format
//  EXPECT_NO_THROW(ReorderBase::Permute2DColWise(c_reorder_vector,
//  &global_csr, {&cpu_context}));
//  EXPECT_EQ((ReorderBase::Permute2DColWise(c_reorder_vector,
//  &global_csr, {&cpu_context}))->get_id(), (format::CSR<int, int,
//  int>::get_id_static())); transformed_format =
//  ReorderBase::Permute2DColWise(c_reorder_vector,
//  &global_csr, {&cpu_context})->Convert<format::CSR>(); for (int i = 0; i <
//  n+1; i++){
//    EXPECT_EQ(transformed_format->get_row_ptr()[i], c_row_ptr[i]);
//  }
//  for (int i = 0; i < nnz; i++){
//    EXPECT_EQ(transformed_format->get_col()[i], c_cols[i]);
//    EXPECT_EQ(transformed_format->get_vals()[i], c_vals[i]);
//  }
//  // converting output to illegal format (No conversion available)
//  EXPECT_THROW(ReorderBase::Permute2DColWise<TestFormat>(c_reorder_vector,
//  &global_csr, {&cpu_context}), utils::ConversionException);
//  // passing a format that isn't convertable
//  TestFormat<int, int, int> f;
//  EXPECT_THROW(ReorderBase::Permute2DColWise<TestFormat>(c_reorder_vector,
//  &f, {&cpu_context}), utils::FunctionNotFoundException);
//}

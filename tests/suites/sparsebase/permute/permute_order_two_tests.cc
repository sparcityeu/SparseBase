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
#include "sparsebase/permute/permute_order_one.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";


using namespace sparsebase;
using namespace sparsebase::preprocess;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::permute;
#include "../functionality_common.inc"
TEST(PermuteTest, RowWise) {
PermuteOrderTwo<int, int, int> transformer(
    r_reorder_vector, nullptr);
EXPECT_THROW(transformer.GetPermutation(&global_coo, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
auto transformed_format =
    transformer.GetPermutation(&global_coo, {&cpu_context}, true)
        ->As<format::CSR>();
for (int i = 0; i < n + 1; i++) {
EXPECT_EQ(transformed_format->get_row_ptr()[i], r_row_ptr[i]);
}
for (int i = 0; i < nnz; i++) {
EXPECT_EQ(transformed_format->get_col()[i], r_cols[i]);
EXPECT_EQ(transformed_format->get_vals()[i], r_vals[i]);
}
}

TEST(InversePermuteTest, RowColWise) {
sparsebase::format::CSR<int, int, int> rc_reordered_csr(
    3, 3, rc_row_ptr, rc_cols, rc_vals, sparsebase::format::kNotOwned, true);
auto inv_r_order = ReorderBase::InversePermutation(
    r_reorder_vector, rc_reordered_csr.get_dimensions()[0]);
auto inv_c_order = ReorderBase::InversePermutation(
    c_reorder_vector, rc_reordered_csr.get_dimensions()[0]);
PermuteOrderTwo<int, int, int> transformer(
    inv_r_order, inv_c_order);
auto transformed_format =
    transformer.GetPermutation(&rc_reordered_csr, {&cpu_context}, true)
        ->As<format::CSR>();
for (int i = 0; i < n + 1; i++) {
EXPECT_EQ(transformed_format->get_row_ptr()[i], row_ptr[i]);
}
for (int i = 0; i < nnz; i++) {
EXPECT_EQ(transformed_format->get_col()[i], cols[i]);
EXPECT_EQ(transformed_format->get_vals()[i], vals[i]);
}
}

// TEST(PermuteTest, ColWise) {
//  PermuteOrderTwo<int, int, int> transformer(nullptr,
//  c_reorder_vector); auto transformed_format =
//      transformer.GetPermutation(&global_coo, {&cpu_context})
//          ->AsAbsolute<format::CSR<int, int, int>>();
//  for (int i = 0; i < n+1; i++){
//    EXPECT_EQ(transformed_format->get_row_ptr()[i], c_row_ptr[i]);
//  }
//  for (int i = 0; i < nnz; i++){
//    EXPECT_EQ(transformed_format->get_col()[i], c_cols[i]);
//    EXPECT_EQ(transformed_format->get_vals()[i], c_vals[i]);
//  }
//}

TEST(PermuteTest, RowColWise) {
PermuteOrderTwo<int, int, int> transformer(
    r_reorder_vector, c_reorder_vector);
auto transformed_format =
    transformer.GetPermutation(&global_coo, {&cpu_context}, true)
        ->As<format::CSR>();
for (int i = 0; i < n + 1; i++) {
EXPECT_EQ(transformed_format->get_row_ptr()[i], rc_row_ptr[i]);
}
for (int i = 0; i < nnz; i++) {
EXPECT_EQ(transformed_format->get_col()[i], rc_cols[i]);
EXPECT_EQ(transformed_format->get_vals()[i], rc_vals[i]);
}
}

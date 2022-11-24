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
#include "sparsebase/permute/permuter.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";


using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::permute;
#include "../functionality_common.inc"


TEST(PermuteTest, ConversionNoParam) {
DegreeReorder<int, int, int> reorder(false);
auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
PermuteOrderTwo<int, int, int> transformer(order,
                                           order);
auto transformed_format =
    transformer.GetPermutation(&global_coo, {&cpu_context}, true)
        ->As<format::CSR>();
confirm_renumbered_csr(
    global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
    global_csr.get_col(), transformed_format->get_col(), order, n);
EXPECT_THROW(transformer.GetPermutation(&global_coo, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
}
TEST(PermuteTest, WrongInputType) {
DegreeReorder<int, int, int> reorder(false);
auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
PermuteOrderTwo<int, int, int> transformer(order,
                                           order);
EXPECT_THROW((transformer.GetPermutation(&orig_arr, {&cpu_context}, true)
->As<format::CSR>()),
utils::FunctionNotFoundException);
}

TEST(PermuteTest, NoConversionParam) {
DegreeReorder<int, int, int> reorder(false);
auto order = reorder.GetReorder(&global_csr, {&cpu_context}, true);
PermuteOrderTwo<int, int, int> transformer(nullptr,
                                           nullptr);
PermuteOrderTwoParams<int> params(order, order);
auto transformed_format =
    transformer.GetPermutation(&global_csr, &params, {&cpu_context}, true)
        ->As<format::CSR>();
confirm_renumbered_csr(
    global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
    global_csr.get_col(), transformed_format->get_col(), order, n);
EXPECT_NO_THROW((transformer.GetPermutation(&global_csr, &params, {&cpu_context}, true)
->As<format::CSR>()));
}

TEST(PermuteTest, ConversionParamCached) {
DegreeReorder<int, int, int> reorder(false);
auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
PermuteOrderTwo<int, int, int> transformer(nullptr,
                                           nullptr);
PermuteOrderTwoParams<int> params(order, order);
auto transformed_output = transformer.GetPermutationCached(
    &global_coo, &params, {&cpu_context}, true);
auto transformed_format = std::get<1>(transformed_output)->As<format::CSR>();
confirm_renumbered_csr(
    global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
    global_csr.get_col(), transformed_format->get_col(), order, n);
EXPECT_EQ(std::get<0>(transformed_output).size(), 1);
EXPECT_EQ(std::get<0>(transformed_output)[0].size(), 1);
ASSERT_NE(std::get<0>(transformed_output)[0][0], nullptr);
auto cached_format = std::get<0>(transformed_output)[0][0]
                         ->AsAbsolute<format::CSR<int, int, int>>();
compare_csr(&global_csr, cached_format);
}

TEST(PermuteTest, NoConversionNoParamCached) {
DegreeReorder<int, int, int> reorder(false);
auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
PermuteOrderTwo<int, int, int> transformer(nullptr,
                                           nullptr);
PermuteOrderTwoParams<int> params(order, order);
auto transformed_output = transformer.GetPermutationCached(
    &global_csr, &params, {&cpu_context}, true);
auto transformed_format = std::get<1>(transformed_output)->As<format::CSR>();
confirm_renumbered_csr(
    global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
    global_csr.get_col(), transformed_format->get_col(), order, n);
EXPECT_EQ(std::get<0>(transformed_output).size(), 1);
ASSERT_EQ(std::get<0>(transformed_output)[0].size(), 0);
}

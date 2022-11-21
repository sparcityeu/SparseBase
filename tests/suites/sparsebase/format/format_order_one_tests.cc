#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/array.h"
#include "common.inc"
TEST(FormatOrderOne, Convert) {
sparsebase::format::Array<int> array(4, coo_vals,
                                     sparsebase::format::kNotOwned);

sparsebase::context::CPUContext cpu_context;
sparsebase::format::Array<int> *conv_arr =
    array.Convert<sparsebase::format::Array>(&cpu_context);
EXPECT_EQ(conv_arr, &array);
// Check the dimensions
EXPECT_EQ(conv_arr->get_num_nnz(), 4);
std::vector<sparsebase::format::DimensionType> expected_dimensions{4};
EXPECT_EQ(conv_arr->get_dimensions(), expected_dimensions);

// Check the array
for (int i = 0; i < 4; i++) {
EXPECT_EQ(conv_arr->get_vals()[i], coo_vals[i]);
}
}

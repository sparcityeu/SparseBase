#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/format/array.h"
#include "common.inc"
TEST(Array, Basics) {
sparsebase::format::Array<int> array(4, coo_vals,
                                     sparsebase::format::kNotOwned);

// Check the dimensions
EXPECT_EQ(array.get_num_nnz(), 4);
std::vector<sparsebase::format::DimensionType> expected_dimensions{4};
EXPECT_EQ(array.get_dimensions(), expected_dimensions);

// Check the array
for (int i = 0; i < 4; i++) {
EXPECT_EQ(array.get_vals()[i], coo_vals[i]);
}
}
TEST(Array, Ownership) {
// Ownership model is designed to work with dynamic memory
int *data = new int[4]{1, 3, 2, 4};

// Construct the Arrays
auto *array_owned =
    new sparsebase::format::Array<int>(4, data, sparsebase::format::kOwned);
auto *array_not_owned = new sparsebase::format::Array<int>(
    4, data, sparsebase::format::kNotOwned);

// Deleting both should not cause an issue since only one should deallocate
// the memory
delete array_owned;
delete array_not_owned;
}
TEST(Array, Release) {
// Ownership model is designed to work with dynamic memory
int *data = new int[4]{1, 3, 2, 4};

// Construct an owned array
auto *array_owned =
    new sparsebase::format::Array<int>(4, data, sparsebase::format::kOwned);

// Release the internal array
auto *data2 = array_owned->release_vals();

// Deleting the Array should not deallocate the internal data
delete array_owned;

// To check delete it manually
delete[] data;
}

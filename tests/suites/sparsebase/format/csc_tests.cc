#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/format/csc.h"
#include "common.inc"
TEST(CSC, Basics) {
// Construct the CSR
sparsebase::format::CSC<int, int, int> csc(4, 4, csc_col_ptr, csc_row,
                                           csc_vals);

// Check the dimensions
EXPECT_EQ(csc.get_num_nnz(), 4);
std::vector<sparsebase::format::DimensionType> expected_dimensions{4, 4};
EXPECT_EQ(csc.get_dimensions(), expected_dimensions);

// Check the arrays
for (int i = 0; i < 4; i++) {
EXPECT_EQ(csc.get_row()[i], csc_row[i]);
EXPECT_EQ(csc.get_vals()[i], csc_vals[i]);
}

for (int i = 0; i < 5; i++) {
EXPECT_EQ(csc.get_col_ptr()[i], csc_col_ptr[i]);
}
}

TEST(CSC, Ownership) {
// Ownership model is designed to work with dynamic memory
// So we copy our static arrays to dynamic ones
// If static arrays are to be used, kNotOwned should always be used
int *new_csc_col_ptr = new int[5];
int *new_csc_row = new int[4];
int *new_csc_vals = new int[4];
std::copy(csc_col_ptr, csc_col_ptr + 5, new_csc_col_ptr);
std::copy(csc_row, csc_row + 4, new_csc_row);
std::copy(csc_vals, csc_vals + 4, new_csc_vals);

// Construct the cscs
auto *csc_owned = new sparsebase::format::CSC<int, int, int>(
    4, 4, new_csc_col_ptr, new_csc_row, new_csc_vals,
    sparsebase::format::kOwned);
auto *csc_not_owned = new sparsebase::format::CSC<int, int, int>(
    4, 4, new_csc_col_ptr, new_csc_row, new_csc_vals,
    sparsebase::format::kNotOwned);

// Deleting both should not cause an issue since only one should deallocate
// the memory
delete csc_not_owned;
delete csc_owned;
}
TEST(CSC, Release) {
// Ownership model is designed to work with dynamic memory
// So we copy our static arrays to dynamic ones
// If static arrays are to be used, kNotOwned should always be used
int *new_csc_col_ptr = new int[5];
int *new_csc_row = new int[4];
int *new_csc_vals = new int[4];
std::copy(csc_col_ptr, csc_col_ptr + 5, new_csc_col_ptr);
std::copy(csc_row, csc_row + 4, new_csc_row);
std::copy(csc_vals, csc_vals + 4, new_csc_vals);

// Construct an owned CSC
auto *csc_owned = new sparsebase::format::CSC<int, int, int>(
    4, 4, new_csc_col_ptr, new_csc_row, new_csc_vals,
    sparsebase::format::kOwned);

// Release the arrays
auto *col_ptr = csc_owned->release_col_ptr();
auto *row = csc_owned->release_row();
auto *vals = csc_owned->release_vals();

// Deleting the CSC should not deallocate arrays
delete csc_owned;

// To check delete them manually
delete[] col_ptr;
delete[] row;
delete[] vals;
}

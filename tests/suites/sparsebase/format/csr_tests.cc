#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/format/csr.h"
#include "common.inc"

TEST(CSR, Basics) {
// Construct the CSR
sparsebase::format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col,
                                           csr_vals);

// Check the dimensions
EXPECT_EQ(csr.get_num_nnz(), 4);
std::vector<sparsebase::format::DimensionType> expected_dimensions{4, 4};
EXPECT_EQ(csr.get_dimensions(), expected_dimensions);

// Check the arrays
for (int i = 0; i < 4; i++) {
EXPECT_EQ(csr.get_col()[i], csr_col[i]);
EXPECT_EQ(csr.get_vals()[i], csr_vals[i]);
}

for (int i = 0; i < 5; i++) {
EXPECT_EQ(csr.get_row_ptr()[i], csr_row_ptr[i]);
}
}
TEST(CSR, Ownership) {
// Ownership model is designed to work with dynamic memory
// So we copy our static arrays to dynamic ones
// If static arrays are to be used, kNotOwned should always be used
int *new_csr_row_ptr = new int[5];
int *new_csr_col = new int[4];
int *new_csr_vals = new int[4];
std::copy(csr_row_ptr, csr_row_ptr + 5, new_csr_row_ptr);
std::copy(csr_col, csr_col + 4, new_csr_col);
std::copy(csr_vals, csr_vals + 4, new_csr_vals);

// Construct the CSRs
auto *csr_owned = new sparsebase::format::CSR<int, int, int>(
    4, 4, new_csr_row_ptr, new_csr_col, new_csr_vals,
    sparsebase::format::kOwned);
auto *csr_not_owned = new sparsebase::format::CSR<int, int, int>(
    4, 4, new_csr_row_ptr, new_csr_col, new_csr_vals,
    sparsebase::format::kNotOwned);

// Deleting both should not cause an issue since only one should deallocate
// the memory
delete csr_not_owned;
delete csr_owned;
}
TEST(CSR, Release) {
// Ownership model is designed to work with dynamic memory
// So we copy our static arrays to dynamic ones
// If static arrays are to be used, kNotOwned should always be used
int *new_csr_row_ptr = new int[5];
int *new_csr_col = new int[4];
int *new_csr_vals = new int[4];
std::copy(csr_row_ptr, csr_row_ptr + 5, new_csr_row_ptr);
std::copy(csr_col, csr_col + 4, new_csr_col);
std::copy(csr_vals, csr_vals + 4, new_csr_vals);

// Construct an owned CSR
auto *csr_owned = new sparsebase::format::CSR<int, int, int>(
    4, 4, new_csr_row_ptr, new_csr_col, new_csr_vals,
    sparsebase::format::kOwned);

// Release the arrays
auto *row_ptr = csr_owned->release_row_ptr();
auto *col = csr_owned->release_col();
auto *vals = csr_owned->release_vals();

// Deleting the CSR should not deallocate arrays
delete csr_owned;

// To check delete them manually
delete[] row_ptr;
delete[] col;
delete[] vals;
}
TEST(CSR, Sort) {
int csr_row_ptr_shuffled[5]{0, 2, 3, 3, 4};
int csr_col_shuffled[4]{2, 0, 1, 3};
int csr_vals_shuffled[4]{5, 4, 7, 9};
sparsebase::format::CSR<int, int, int> csr(
    4, 4, csr_row_ptr_shuffled, csr_col_shuffled, csr_vals_shuffled,
    sparsebase::format::kNotOwned);

for (int i = 0; i < 4; i++) {
EXPECT_EQ(csr.get_col()[i], csr_col[i]);
EXPECT_EQ(csr.get_vals()[i], csr_vals[i]);
}

int csr_row_ptr_shuffled2[5]{0, 2, 3, 3, 4};
int csr_col_shuffled2[4]{2, 0, 1, 3};
int csr_vals_shuffled2[4]{5, 4, 7, 9};
sparsebase::format::CSR<int, int, int> csr2(
    4, 4, csr_row_ptr_shuffled2, csr_col_shuffled2, csr_vals_shuffled2,
    sparsebase::format::kNotOwned, true);

for (int i = 0; i < 4; i++) {
EXPECT_EQ(csr2.get_col()[i], csr_col_shuffled2[i]);
EXPECT_EQ(csr2.get_vals()[i], csr_vals_shuffled2[i]);
}

int csr_row_ptr_shuffled3[5]{0, 2, 3, 3, 4};
int csr_col_shuffled3[4]{2, 0, 1, 3};
sparsebase::format::CSR<int, int, void> csr3(4, 4, csr_row_ptr_shuffled3,
                                             csr_col_shuffled3, nullptr,
                                             sparsebase::format::kNotOwned);

EXPECT_EQ(csr3.get_vals(), nullptr);
for (int i = 0; i < 4; i++) {
EXPECT_EQ(csr3.get_col()[i], csr_col[i]);
}
}

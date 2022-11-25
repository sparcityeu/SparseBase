#include <iostream>

#include "common.inc"
#include "gtest/gtest.h"
#include "sparsebase/format/coo.h"

TEST(COO, Basics) {
  sparsebase::format::COO<int, int, int> coo(4, 4, 4, coo_row, coo_col,
                                             coo_vals);

  // Check the dimensions
  EXPECT_EQ(coo.get_num_nnz(), 4);
  std::vector<sparsebase::format::DimensionType> expected_dimensions{4, 4};
  EXPECT_EQ(coo.get_dimensions(), expected_dimensions);

  // Check the arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo.get_row()[i], coo_row[i]);
    EXPECT_EQ(coo.get_col()[i], coo_col[i]);
    EXPECT_EQ(coo.get_vals()[i], coo_vals[i]);
  }
}

TEST(COO, Ownership) {
  // Ownership model is designed to work with dynamic memory
  // So we copy our static arrays to dynamic ones
  // If static arrays are to be used, kNotOwned should always be used
  int *new_coo_row = new int[4];
  int *new_coo_col = new int[4];
  int *new_coo_vals = new int[4];
  std::copy(coo_row, coo_row + 4, new_coo_row);
  std::copy(coo_col, coo_col + 4, new_coo_col);
  std::copy(coo_vals, coo_vals + 4, new_coo_vals);

  // Construct the CSRs
  auto *coo_owned = new sparsebase::format::COO<int, int, int>(
      4, 4, 4, new_coo_row, new_coo_col, new_coo_vals,
      sparsebase::format::kOwned);
  auto *coo_not_owned = new sparsebase::format::COO<int, int, int>(
      4, 4, 4, new_coo_row, new_coo_col, new_coo_vals,
      sparsebase::format::kNotOwned);

  // Deleting both should not cause an issue since only one should deallocate
  // the memory
  delete coo_not_owned;
  delete coo_owned;
}
TEST(COO, Release) {
  // Ownership model is designed to work with dynamic memory
  // So we copy our static arrays to dynamic ones
  // If static arrays are to be used, kNotOwned should always be used
  int *new_coo_row = new int[4];
  int *new_coo_col = new int[4];
  int *new_coo_vals = new int[4];
  std::copy(coo_row, coo_row + 4, new_coo_row);
  std::copy(coo_col, coo_col + 4, new_coo_col);
  std::copy(coo_vals, coo_vals + 4, new_coo_vals);

  // Construct an owned COO
  auto *coo_owned = new sparsebase::format::COO<int, int, int>(
      4, 4, 4, new_coo_row, new_coo_col, new_coo_vals,
      sparsebase::format::kOwned);

  // Release the arrays
  auto *row = coo_owned->release_row();
  auto *col = coo_owned->release_col();
  auto *vals = coo_owned->release_vals();

  // Deleting the COO should not deallocate arrays
  delete coo_owned;

  // To check delete them manually
  delete[] row;
  delete[] col;
  delete[] vals;
}
TEST(COO, Sort) {
  int coo_row_shuffled[4]{0, 0, 3, 1};
  int coo_col_shuffled[4]{2, 0, 3, 1};
  int coo_vals_shuffled[4]{5, 4, 9, 7};
  sparsebase::format::COO<int, int, int> coo(
      4, 4, 4, coo_row_shuffled, coo_col_shuffled, coo_vals_shuffled,
      sparsebase::format::kNotOwned);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo.get_row()[i], coo_row[i]);
    EXPECT_EQ(coo.get_col()[i], coo_col[i]);
    EXPECT_EQ(coo.get_vals()[i], coo_vals[i]);
  }

  int coo_row_shuffled2[4]{0, 0, 3, 1};
  int coo_col_shuffled2[4]{2, 0, 3, 1};
  int coo_vals_shuffled2[4]{5, 4, 9, 7};
  sparsebase::format::COO<int, int, int> coo2(
      4, 4, 4, coo_row_shuffled2, coo_col_shuffled2, coo_vals_shuffled2,
      sparsebase::format::kNotOwned, true);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo2.get_row()[i], coo_row_shuffled2[i]);
    EXPECT_EQ(coo2.get_col()[i], coo_col_shuffled2[i]);
    EXPECT_EQ(coo2.get_vals()[i], coo_vals_shuffled2[i]);
  }

  int coo_row_shuffled3[4]{0, 0, 3, 1};
  int coo_col_shuffled3[4]{2, 0, 3, 1};
  sparsebase::format::COO<int, int, void> coo3(4, 4, 4, coo_row_shuffled3,
                                               coo_col_shuffled3, nullptr,
                                               sparsebase::format::kNotOwned);

  EXPECT_EQ(coo3.get_vals(), nullptr);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo3.get_row()[i], coo_row[i]);
    EXPECT_EQ(coo3.get_col()[i], coo_col[i]);
  }
}

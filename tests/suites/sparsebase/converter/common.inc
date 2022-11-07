#include "gtest/gtest.h"
// The arrays defined here are for two matrices
// One in csr format one in coo format
// These are known to be equivalent (converted using scipy)
const int n = 12;
const int m = 9;
const int nnz = 7;
int coo_row[7]{0, 0, 1, 3, 5, 10, 11};
int coo_col[7]{0, 2, 1, 3, 3, 8, 7};
int coo_vals[7]{3, 5, 7, 9, 15, 11, 13};
int csr_row_ptr[13]{0, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7};
int csr_col[7]{0, 2, 1, 3, 3, 8, 7};
int csr_vals[7]{3, 5, 7, 9, 15, 11, 13};
int csc_col_ptr[13]{0, 1, 2, 3, 5, 5, 5, 5, 6, 7, 7, 7, 7};
int csc_row[7]{0, 1, 0, 3, 5, 11, 10};
int csc_vals[7]{3, 7, 5, 9, 15, 13, 11};

template <typename T1, typename T2>
void compare_arrays(T1* a1, T2* a2, int size, std::string arr_name) {
  for (int i = 0; i < size; i++)
    EXPECT_EQ(a1[i], a2[i]) << "problem in array " + arr_name + " index ",
        std::to_string(i);
}
template <typename CSRType1, typename CSRType2>
void compare_csrs(CSRType1* csr1, CSRType2* csr2) {
  auto nnz1 = csr1->get_num_nnz();
  auto nnz2 = csr2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csr1->get_dimensions()[0], csr2->get_dimensions()[0]);
  ASSERT_EQ(csr1->get_dimensions()[1], csr2->get_dimensions()[1]);
  compare_arrays(csr1->get_row_ptr(), csr2->get_row_ptr(), n + 1, "row_ptr");
  compare_arrays(csr1->get_col(), csr2->get_col(), nnz1, "col");
  compare_arrays(csr1->get_vals(), csr2->get_vals(), nnz1, "vals");
}

template <typename COOType1, typename COOType2>
void compare_coos(COOType1* coo1, COOType2* coo2) {
  auto nnz1 = coo1->get_num_nnz();
  auto nnz2 = coo2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(coo1->get_dimensions()[0], coo2->get_dimensions()[0]);
  ASSERT_EQ(coo1->get_dimensions()[1], coo2->get_dimensions()[1]);
  compare_arrays(coo1->get_row(), coo2->get_row(), nnz1, "row");
  compare_arrays(coo1->get_col(), coo2->get_col(), nnz1, "col");
  compare_arrays(coo1->get_vals(), coo2->get_vals(), nnz1, "vals");
}

template <typename CSCType1, typename CSCType2>
void compare_cscs(CSCType1* csc1, CSCType2* csc2) {
  auto nnz1 = csc1->get_num_nnz();
  auto nnz2 = csc2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csc1->get_dimensions()[0], csc2->get_dimensions()[0]);
  ASSERT_EQ(csc1->get_dimensions()[1], csc2->get_dimensions()[1]);
  compare_arrays(csc1->get_col_ptr(), csc2->get_col_ptr(), n + 1, "col_ptr");
  compare_arrays(csc1->get_row(), csc2->get_row(), nnz1, "row");
  compare_arrays(csc1->get_vals(), csc2->get_vals(), nnz1, "vals");
}


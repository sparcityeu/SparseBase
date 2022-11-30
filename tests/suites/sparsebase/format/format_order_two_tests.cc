#include <iostream>

#include "common.inc"
#include "gtest/gtest.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_two.h"

TEST(FormatOrderTwo, ConvertSameFormat) {
  // Construct the CSR
  sparsebase::format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col,
                                             csr_vals);

  sparsebase::context::CPUContext cpu_context;
  sparsebase::format::CSR<int, int, int> *csr_converted =
      csr.Convert<sparsebase::format::CSR>(&cpu_context);
  EXPECT_EQ(csr_converted, &csr);
  // Check the dimensions
  EXPECT_EQ(csr_converted->get_num_nnz(), 4);
  std::vector<sparsebase::format::DimensionType> expected_dimensions{4, 4};
  EXPECT_EQ(csr_converted->get_dimensions(), expected_dimensions);

  // Check the arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr_converted->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr_converted->get_vals()[i], csr_vals[i]);
  }

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr_converted->get_row_ptr()[i], csr_row_ptr[i]);
  }
}

TEST(FormatOrderTwo, ConvertDifferentFormat) {
  // Construct the CSR
  sparsebase::format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col,
                                             csr_vals);

  sparsebase::context::CPUContext cpu_context;
  sparsebase::format::COO<int, int, int> *coo_converted =
      csr.Convert<sparsebase::format::COO>(&cpu_context);
  // Check the dimensions
  EXPECT_EQ(coo_converted->get_num_nnz(), 4);
  std::vector<sparsebase::format::DimensionType> expected_dimensions{4, 4};
  EXPECT_EQ(coo_converted->get_dimensions(), expected_dimensions);

  // Check the arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo_converted->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo_converted->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo_converted->get_vals()[i], coo_vals[i]);
  }
}


template <typename T>
class dummy{};


TEST(Is, FormatOrderTwo){
  
  sparsebase::format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col,
                                             csr_vals);
  sparsebase::format::COO<int, int, int> coo(4, 4, 4, coo_row, coo_col,
                                             coo_vals);
  EXPECT_THROW((csr.AsAbsolute<sparsebase::format::COO<int, int, int>>()), sparsebase::utils::TypeException);
  EXPECT_ANY_THROW((csr.AsAbsolute<sparsebase::format::COO<int, int, int>>()));
  EXPECT_EQ(csr.Is<sparsebase::format::CSR>(), true);
  EXPECT_TRUE(csr.Is<sparsebase::format::CSR>());
  EXPECT_FALSE(csr.Is<sparsebase::format::COO>());
  EXPECT_FALSE(csr.Is<sparsebase::format::FormatOrderTwo>());
}

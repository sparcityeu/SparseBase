#include "sparsebase/sparsebase.h"
#include "gtest/gtest.h"
#include <iostream>

// The arrays defined here are for two matrices
// One in csr format one in coo format
// These are known to be equivalent (converted using scipy)
const int n = 12;
const int m = 9;
const int nnz = 7;
int coo_row[7] {0, 0, 1, 3, 5, 10, 11};
int coo_col[7] {0, 2, 1, 3, 3, 8, 7};
int coo_vals[7]{3, 5, 7, 9, 15, 11, 13};
int csr_row_ptr[13]{0, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7};
int csr_col[7]{0, 2, 1, 3, 3, 8, 7};
int csr_vals[7]{3, 5, 7, 9, 15, 11, 13};
int csc_col_ptr[13]{0, 1, 2, 3, 5, 5, 5, 5, 6, 7, 7, 7, 7};
int csc_row[7]{0, 1, 0, 3, 5, 11, 10};
int csc_vals[7]{3, 7, 5, 9, 15, 13, 11};

TEST(ConverterOrderTwo, CSRToCOO) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Testing non-move converter (deep copy)
  std::cout << "Testing non-move converter (deep copy)" << std::endl;
  auto coo = converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(
      &csr, &cpu_context, false);

  // None of the pointers should be the same due to deep copy
  EXPECT_NE(coo->get_row(), csr.get_row_ptr());
  EXPECT_NE(coo->get_col(), csr.get_col());
  EXPECT_NE(coo->get_vals(), csr.get_vals());

  // All values should be equal however
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(coo->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo->get_vals()[i], coo_vals[i]);
  }

  // Testing move converter (some arrays can be shallow copied)
  std::cout << "Testing move converter" << std::endl;
  auto coo2 = converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(
      &csr, &cpu_context, true);

  // Particularly, these two arrays should have been moved
  EXPECT_EQ(coo2->get_col(), csr.get_col());
  EXPECT_EQ(coo2->get_vals(), csr.get_vals());

  // row array should be constructed from scratch
  EXPECT_NE(coo2->get_row(), csr.get_row_ptr());

  // All values should be equal
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(coo2->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo2->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo2->get_vals()[i], coo_vals[i]);
  }

  std::cout << "End of test" << std::endl;
}

TEST(ConverterOrderTwo, CSRToCSC) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Testing non-move converter (deep copy)
  std::cout << "Testing non-move converter (deep copy)" << std::endl;
  auto csc = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &csr, &cpu_context, false);

  // None of the pointers should be the same due to deep copy
  EXPECT_NE(csc->get_row(), csr.get_row_ptr());
  EXPECT_NE(csc->get_col_ptr(), csr.get_col());
  EXPECT_NE(csc->get_vals(), csr.get_vals());

  // All values should be equal however
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csc->get_row()[i], csc_row[i]);
    EXPECT_EQ(csc->get_col_ptr()[i], csc_col_ptr[i]);
    EXPECT_EQ(csc->get_vals()[i], csc_vals[i]);
  }

  // Testing move converter (some arrays can be shallow copied)
  std::cout << "Testing move converter" << std::endl;
  auto csc2 = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &csr, &cpu_context, true);

  // Particularly, these two arrays should have been moved
  EXPECT_NE(csc2->get_col_ptr(), csr.get_col());
  EXPECT_NE(csc2->get_vals(), csr.get_vals());

  // row array should be constructed from scratch
  EXPECT_NE(csc2->get_row(), csr.get_row_ptr());

  // All values should be equal
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csc2->get_row()[i], csc_row[i]);
    EXPECT_EQ(csc2->get_col_ptr()[i], csc_col_ptr[i]);
    EXPECT_EQ(csc2->get_vals()[i], csc_vals[i]);
  }

  std::cout << "End of test" << std::endl;
}

TEST(ConverterOrderTwo, COOToCSR) {
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Testing non-move converter (deep copy)
  auto csr = converterOrderTwo.Convert<sparsebase::format::CSR<int, int, int>>(
      &coo, &cpu_context, false);

  // None of the pointers should be the same due to deep copy
  EXPECT_NE(csr->get_row_ptr(), coo.get_row());
  EXPECT_NE(csr->get_col(), coo.get_col());
  EXPECT_NE(csr->get_vals(), coo.get_vals());

  // All values should be equal however
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csr->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr->get_vals()[i], csr_vals[i]);
  }

  // All values should be equal however
  for (int i = 0; i < (n + 1); i++) {
    EXPECT_EQ(csr->get_row_ptr()[i], csr_row_ptr[i]);
  }

  // Testing move converter (some arrays can be shallow copied)
  auto csr2 = converterOrderTwo.Convert<sparsebase::format::CSR<int, int, int>>(
      &coo, &cpu_context, true);

  // Particularly, these two arrays should have been moved
  EXPECT_EQ(csr2->get_col(), coo.get_col());
  EXPECT_EQ(csr2->get_vals(), coo.get_vals());

  // row_ptr array should be constructed from scratch
  EXPECT_NE(csr2->get_row_ptr(), coo.get_row());

  // All values should be equal
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csr2->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr2->get_vals()[i], csr_vals[i]);
  }

  // All values should be equal
  for (int i = 0; i < (n + 1); i++) {
    EXPECT_EQ(csr2->get_row_ptr()[i], csr_row_ptr[i]);
  }
}

TEST(ConverterOrderTwo, COOToCSC) {
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Testing non-move converter (deep copy)
  auto csc = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &coo, &cpu_context, false);

  // None of the pointers should be the same due to deep copy
  EXPECT_NE(csc->get_col_ptr(), coo.get_col());
  EXPECT_NE(csc->get_row(), coo.get_row());
  EXPECT_NE(csc->get_vals(), coo.get_vals());

  // All values should be equal however
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csc->get_row()[i], csc_row[i]);
    EXPECT_EQ(csc->get_vals()[i], csc_vals[i]);
  }

  // All values should be equal however
  for (int i = 0; i < (n + 1); i++) {
    EXPECT_EQ(csc->get_col_ptr()[i], csc_col_ptr[i]);
  }

  // Testing move converter (no arrays can be shallow copied)
  auto csc2 = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &coo, &cpu_context, true);

  // Particularly, these two arrays should have been moved
  EXPECT_NE(csc2->get_row(), coo.get_row());
  EXPECT_NE(csc2->get_vals(), coo.get_vals());

  // row_ptr array should be constructed from scratch
  EXPECT_NE(csc2->get_col_ptr(), coo.get_col());

  // All values should be equal
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csc2->get_row()[i], csc_row[i]);
    EXPECT_EQ(csc2->get_vals()[i], csc_vals[i]);
  }

  // All values should be equal
  for (int i = 0; i < (n + 1); i++) {
    EXPECT_EQ(csc2->get_col_ptr()[i], csc_col_ptr[i]);
  }
}

TEST(ConverterOrderTwo, COOToCOO) {
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Self conversion should do nothing
  auto coo2 = converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(
      &coo, &cpu_context);

  // As a result the pointers should not change
  EXPECT_EQ(coo2->get_row(), coo.get_row());
  EXPECT_EQ(coo2->get_col(), coo.get_col());
  EXPECT_EQ(coo2->get_vals(), coo.get_vals());

  // And values should also not change
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(coo2->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo2->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo2->get_vals()[i], coo_vals[i]);
  }
}

TEST(ConverterOrderTwo, CSRToCSR) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Self conversion should do nothing
  auto csr2 = converterOrderTwo.Convert<sparsebase::format::CSR<int, int, int>>(
      &csr, &cpu_context);

  // As a result the pointers should not change
  EXPECT_EQ(csr2->get_row_ptr(), csr.get_row_ptr());
  EXPECT_EQ(csr2->get_col(), csr.get_col());
  EXPECT_EQ(csr2->get_vals(), csr.get_vals());

  // And values should also not change
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(csr2->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr2->get_vals()[i], csr_vals[i]);
  }

  // And values should also not change
  for (int i = 0; i < (n + 1); i++) {
    EXPECT_EQ(csr2->get_row_ptr()[i], csr_row_ptr[i]);
  }
}

TEST(Converter, ClearingAllFunctions){
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  converterOrderTwo.ClearConversionFunctions();
  EXPECT_THROW((converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(&csr, &cpu_context)), sparsebase::utils::ConversionException);
  converterOrderTwo
      .RegisterConditionalConversionFunction(
          sparsebase::format::CSR<int, int, int>::get_format_id_static(),
          sparsebase::format::COO<int, int, int>::get_format_id_static(),
          [](sparsebase::format::Format *, sparsebase::context::Context*) -> sparsebase::format::Format* {return nullptr;},
          [](sparsebase::context::Context*, sparsebase::context::Context*) -> bool { return true; });
  EXPECT_EQ(
      (converterOrderTwo.Convert(
          &csr, sparsebase::format::COO<int, int, int>::get_format_id_static(),
          &cpu_context)),
      nullptr);
}

TEST(Converter, ClearingASingleDirection){
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  converterOrderTwo.ClearConversionFunctions(
      sparsebase::format::COO<int, int, int>::get_format_id_static(),
      sparsebase::format::CSR<int, int, int>::get_format_id_static());
  EXPECT_NO_THROW((converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(&csr, &cpu_context)));
  converterOrderTwo.ClearConversionFunctions(
      sparsebase::format::CSR<int, int, int>::get_format_id_static(),
      sparsebase::format::COO<int, int, int>::get_format_id_static());
  EXPECT_THROW((converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(&csr, &cpu_context)), sparsebase::utils::ConversionException);
  converterOrderTwo
      .RegisterConditionalConversionFunction(
          sparsebase::format::CSR<int, int, int>::get_format_id_static(),
          sparsebase::format::COO<int, int, int>::get_format_id_static(),
          [](sparsebase::format::Format *, sparsebase::context::Context*) -> sparsebase::format::Format* {return nullptr;},
          [](sparsebase::context::Context*, sparsebase::context::Context*) -> bool { return true; });
  EXPECT_EQ(
      (converterOrderTwo.Convert(
          &csr, sparsebase::format::COO<int, int, int>::get_format_id_static(),
          &cpu_context)),
      nullptr);
}
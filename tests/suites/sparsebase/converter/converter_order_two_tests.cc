#include <iostream>

#include "common.inc"
#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
using namespace sparsebase;
using namespace converter;

TEST(ConverterOrderTwo, CSRToCOO) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Testing non-move converter (deep copy)
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
}
TEST(ConverterOrderTwo, CSRToCSCMultipleContexts) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> correct_csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context1, cpu_context2;

  // Testing non-move converter (deep copy)

  // Templated call
  auto csc = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &csr, {&cpu_context1, &cpu_context2}, false);

  compare_cscs(csc, &correct_csc);

  // Non-templated call
  csc = converterOrderTwo
            .Convert(&csr, format::CSC<int, int, int>::get_id_static(),
                     {&cpu_context1, &cpu_context2}, false)
            ->AsAbsolute<format::CSC<int, int, int>>();

  compare_cscs(csc, &correct_csc);

  // templated member call
  csc = csr.Convert<format::CSC>({&cpu_context1, &cpu_context2}, false);

  compare_cscs(csc, &correct_csc);
  // remove the function CSR-CSC and try multi-step
  converterOrderTwo.ClearConversionFunctions(
      format::CSC<int, int, int>::get_id_static(),
      format::CSR<int, int, int>::get_id_static(), false);
  converterOrderTwo.ClearConversionFunctions(
      format::CSR<int, int, int>::get_id_static(),
      format::CSC<int, int, int>::get_id_static(), false);
  // Templated call
  csc = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &csr, {&cpu_context1, &cpu_context2}, false);

  compare_cscs(csc, &correct_csc);

  // Non-templated call
  csc = converterOrderTwo
            .Convert(&csr, format::CSC<int, int, int>::get_id_static(),
                     {&cpu_context1, &cpu_context2}, false)
            ->AsAbsolute<format::CSC<int, int, int>>();

  compare_cscs(csc, &correct_csc);

  // templated member call
  csc = csr.Convert<format::CSC>({&cpu_context1, &cpu_context2}, false);
}
TEST(ConverterOrderTwo, CSRToCSCCached) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> correct_csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> correct_coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context1, cpu_context2;

  // Testing non-move converter (deep copy)

  // Non-templated call
  auto cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_id_static(), {&cpu_context1},
      false);
  EXPECT_EQ(cscs.size(), 1);

  // compare_csrs(cscs[0]->AsAbsolute<format::CSR<int, int, int>>(), &csr);

  compare_cscs(cscs[0]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[0];

  // Non-templated call multiple contexts
  cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_id_static(),
      {&cpu_context1, &cpu_context2}, false);
  EXPECT_EQ(cscs.size(), 1);

  // compare_csrs(cscs[0]->AsAbsolute<format::CSR<int, int, int>>(), &csr);

  compare_cscs(cscs[0]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[0];
  // remove the function CSR-CSC and try multi-step
  converterOrderTwo.ClearConversionFunctions(
      format::CSC<int, int, int>::get_id_static(),
      format::CSR<int, int, int>::get_id_static(), false);
  converterOrderTwo.ClearConversionFunctions(
      format::CSR<int, int, int>::get_id_static(),
      format::CSC<int, int, int>::get_id_static(), false);
  // Non-templated call
  cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_id_static(), {&cpu_context1},
      false);
  EXPECT_EQ(cscs.size(), 2);

  compare_cscs(cscs[1]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[1];
  compare_coos(cscs[0]->AsAbsolute<format::COO<int, int, int>>(), &correct_coo);
  delete cscs[0];

  // Non-templated call multiple contexts
  cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_id_static(),
      {&cpu_context1, &cpu_context2}, false);
  EXPECT_EQ(cscs.size(), 2);

  compare_cscs(cscs[1]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[1];
  compare_coos(cscs[0]->AsAbsolute<format::COO<int, int, int>>(), &correct_coo);
  delete cscs[0];
}

TEST(ConverterOrderTwo, CSRToCSC) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  // Testing non-move converter (deep copy)
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
}

TEST(ConverterOrderTwo, COOToCSR) {
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
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
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
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
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
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
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
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

#include "sparsebase/sparsebase.h"
#include "gtest/gtest.h"
#include <iostream>
using namespace sparsebase;
using namespace utils::converter;

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

template <typename T1, typename T2>
void compare_arrays(T1* a1, T2* a2, int size, std::string arr_name){
  for (int i =0; i < size; i++) EXPECT_EQ(a1[i], a2[i]) <<  "problem in array " + arr_name + " index ", std::to_string(i);
}
template <typename CSRType1, typename CSRType2>
void compare_csrs(CSRType1* csr1, CSRType2* csr2){
  auto nnz1 = csr1->get_num_nnz();
  auto nnz2 = csr2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csr1->get_dimensions()[0],csr2->get_dimensions()[0]);
  ASSERT_EQ(csr1->get_dimensions()[1],csr2->get_dimensions()[1]);
  compare_arrays(csr1->get_row_ptr(),csr2->get_row_ptr(), n+1, "row_ptr");
  compare_arrays(csr1->get_col(), csr2->get_col(), nnz1, "col");
  compare_arrays(csr1->get_vals(),csr2->get_vals(), nnz1, "vals");
}

template <typename COOType1, typename COOType2>
void compare_coos(COOType1* coo1, COOType2* coo2){
  auto nnz1 = coo1->get_num_nnz();
  auto nnz2 = coo2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(coo1->get_dimensions()[0],coo2->get_dimensions()[0]);
  ASSERT_EQ(coo1->get_dimensions()[1],coo2->get_dimensions()[1]);
  compare_arrays(coo1->get_row(),coo2->get_row(), nnz1, "row");
  compare_arrays(coo1->get_col(), coo2->get_col(), nnz1, "col");
  compare_arrays(coo1->get_vals(),coo2->get_vals(), nnz1, "vals");
}

template <typename CSCType1, typename CSCType2>
void compare_cscs(CSCType1* csc1, CSCType2* csc2){
  auto nnz1 = csc1->get_num_nnz();
  auto nnz2 = csc2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csc1->get_dimensions()[0],csc2->get_dimensions()[0]);
  ASSERT_EQ(csc1->get_dimensions()[1],csc2->get_dimensions()[1]);
  compare_arrays(csc1->get_col_ptr(),csc2->get_col_ptr(), n+1, "col_ptr");
  compare_arrays(csc1->get_row(), csc2->get_row(), nnz1, "row");
  compare_arrays(csc1->get_vals(),csc2->get_vals(), nnz1, "vals");
}

TEST(ConverterOrderTwo, CSRToCOO) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
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
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context1, cpu_context2;

  // Testing non-move converter (deep copy)

  // Templated call
  auto csc = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &csr, {&cpu_context1, &cpu_context2}, false);

  compare_cscs(csc, &correct_csc);

  // Non-templated call
  csc = converterOrderTwo.Convert(
      &csr, format::CSC<int, int, int>::get_format_id_static(), {&cpu_context1, &cpu_context2}, false)->AsAbsolute<format::CSC<int, int, int>>();

  compare_cscs(csc, &correct_csc);

  //templated member call
  csc = csr.Convert<format::CSC>({&cpu_context1, &cpu_context2}, false);

  compare_cscs(csc, &correct_csc);
  // remove the function CSR-CSC and try multi-step
  converterOrderTwo.ClearConversionFunctions(format::CSC<int, int, int>::get_format_id_static(),
                                             format::CSR<int, int, int>::get_format_id_static(), false);
  converterOrderTwo.ClearConversionFunctions(format::CSR<int, int, int>::get_format_id_static(),
                                             format::CSC<int, int, int>::get_format_id_static(), false);
  // Templated call
  csc = converterOrderTwo.Convert<sparsebase::format::CSC<int, int, int>>(
      &csr, {&cpu_context1, &cpu_context2}, false);

  compare_cscs(csc, &correct_csc);

  // Non-templated call
  csc = converterOrderTwo.Convert(
      &csr, format::CSC<int, int, int>::get_format_id_static(), {&cpu_context1, &cpu_context2}, false)->AsAbsolute<format::CSC<int, int, int>>();

  compare_cscs(csc, &correct_csc);

  //templated member call
  csc = csr.Convert<format::CSC>({&cpu_context1, &cpu_context2}, false);

}
TEST(ConverterOrderTwo, CSRToCSCCached) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> correct_csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> correct_coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
  sparsebase::context::CPUContext cpu_context1, cpu_context2;

  // Testing non-move converter (deep copy)

  // Non-templated call
  auto cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_format_id_static(), {&cpu_context1}, false);
  EXPECT_EQ(cscs.size(), 1);

  //compare_csrs(cscs[0]->AsAbsolute<format::CSR<int, int, int>>(), &csr);

  compare_cscs(cscs[0]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[0];

  // Non-templated call multiple contexts
  cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_format_id_static(), {&cpu_context1, &cpu_context2}, false);
  EXPECT_EQ(cscs.size(), 1);

  //compare_csrs(cscs[0]->AsAbsolute<format::CSR<int, int, int>>(), &csr);

  compare_cscs(cscs[0]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[0];
  // remove the function CSR-CSC and try multi-step
  converterOrderTwo.ClearConversionFunctions(format::CSC<int, int, int>::get_format_id_static(),
                                             format::CSR<int, int, int>::get_format_id_static(), false);
  converterOrderTwo.ClearConversionFunctions(format::CSR<int, int, int>::get_format_id_static(),
                                             format::CSC<int, int, int>::get_format_id_static(), false);
  // Non-templated call
  cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_format_id_static(), {&cpu_context1}, false);
  EXPECT_EQ(cscs.size(), 2);

  compare_cscs(cscs[1]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[1];
  compare_coos(cscs[0]->AsAbsolute<format::COO<int, int, int>>(), &correct_coo);
  delete cscs[0];

  // Non-templated call multiple contexts
  cscs = converterOrderTwo.ConvertCached(
      &csr, format::CSC<int, int, int>::get_format_id_static(), {&cpu_context1, &cpu_context2}, false);
  EXPECT_EQ(cscs.size(), 2);

  compare_cscs(cscs[1]->AsAbsolute<format::CSC<int, int, int>>(), &correct_csc);
  delete cscs[1];
  compare_coos(cscs[0]->AsAbsolute<format::COO<int, int, int>>(), &correct_coo);
  delete cscs[0];
}

TEST(ConverterOrderTwo, CSRToCSC) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int, int, int>
      converterOrderTwo;
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
  converterOrderTwo.RegisterConversionFunction(
      sparsebase::format::CSR<int, int, int>::get_format_id_static(),
      sparsebase::format::COO<int, int, int>::get_format_id_static(),
      [](sparsebase::format::Format *, sparsebase::context::Context *)
          -> sparsebase::format::Format * { return nullptr; },
      [](sparsebase::context::Context *,
         sparsebase::context::Context *) -> bool { return true; });
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
  converterOrderTwo.RegisterConversionFunction(
      sparsebase::format::CSR<int, int, int>::get_format_id_static(),
      sparsebase::format::COO<int, int, int>::get_format_id_static(),
      [](sparsebase::format::Format *, sparsebase::context::Context *)
          -> sparsebase::format::Format * { return nullptr; },
      [](sparsebase::context::Context *,
         sparsebase::context::Context *) -> bool { return true; });
  EXPECT_EQ(
      (converterOrderTwo.Convert(
          &csr, sparsebase::format::COO<int, int, int>::get_format_id_static(),
          &cpu_context)),
      nullptr);
}

#define MATCH_CHECKING_ONEVAL(T1, T2, val) \
  EXPECT_TRUE(utils::isTypeConversionSafe((T1)(val), (T2)(val))); \
  EXPECT_TRUE(utils::isTypeConversionSafe((T2)(val), (T1)(val))); \
  if ((val)!=0) {                           \
    EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-(val)), (T2)(val)));  \
  EXPECT_FALSE(utils::isTypeConversionSafe((T1)((val)), (T2)(-(val)))); \
  EXPECT_FALSE(utils::isTypeConversionSafe((T2)(-(val)), (T1)(val)));  \
  EXPECT_FALSE(utils::isTypeConversionSafe((T2)((val)), (T1)(-(val))));  \
}
#define MATCH_CHECKING_SIGNED_MISMATCH(T1, T2) \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(-1), (T1)(-1))); \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(-1.5), (T1)(-1.5)));

#define MATCH_CHECKING_SIGNED_MATCH(T1, T2) \
  EXPECT_TRUE(format::isTypeConversionSafe((T2)(-1), (T1)(-1))); \
  if constexpr (std::is_integral_v<T1> == std::is_integral_v<T2>) EXPECT_TRUE(format::isTypeConversionSafe((T2)(-2.5), (T1)(-2.5)));

#define MATCH_CHECKING_INT_FLOAT_MISMATCH(T1, T2) \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(-3.5), (T1)(-3.5))); \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(4.5), (T1)(4.5)));

#define MATCH_CHECKING_INT_FLOAT_MATCH(T1, T2) \
  EXPECT_TRUE(format::isTypeConversionSafe((T2)(-5.5), (T1)(-5.5))); \
  EXPECT_TRUE(format::isTypeConversionSafe((T2)(6.5), (T1)(6.5)));

#include <limits>
#include <stdint.h>
#define MATCH_CHECK_ALL_CASES(T1, T2) \
  MATCH_CHECKING_ONEVAL(T1, T2, 0) \
  MATCH_CHECKING_ONEVAL(T1, T2, 1)    \
  if constexpr (std::is_integral_v<T1>) { \
    if constexpr (std::is_signed_v<T1>) {  \
      if constexpr (std::is_integral_v<T2>) {                                 \
        if constexpr (std::is_signed_v<T2>) EXPECT_TRUE(utils::isTypeConversionSafe((T1)(-1), (T2)(-1))); \
        else EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-1), (T2)(-1)));                             \
      } else {                          \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(2), (T2)(2))); \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-2), (T2)(-2))); \
      }                               \
    } else {                          \
        \
    }                                    \
  } else {                            \
    if constexpr (std::is_integral_v<T2>){\
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(1.5), (T2)(1.5)));                                  \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(3), (T2)(3)));                                  \
      if constexpr (std::is_signed_v<T2>){\
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-1.5), (T2)(-1.5)));                                  \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-3), (T2)(-3)));                                  \
      } else {                        \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-3), (T2)(-3)));                                  \
      }                                  \
    }                                     \
  }                                   \
  if constexpr (std::is_floating_point_v<T2> == false) {                                     \
    if constexpr (std::is_integral_v<T1> == true && std::is_integral_v<T2> == true) {                                    \
      if constexpr (std::is_floating_point_v<T1> == false && std::uintmax_t(std::numeric_limits<T1>::max()) > std::uintmax_t(std::numeric_limits<T2>::max())) {  \
      EXPECT_FALSE(utils::isTypeConversionSafe((T1)(std::numeric_limits<T2>::max() + (unsigned long long)1), (T2)((unsigned long long)std::numeric_limits<T2>::max() + (unsigned long long)1)));} \
      {                                   \
        const intmax_t botT1 = std::numeric_limits<T1>::min(); \
        const intmax_t botT2 = std::numeric_limits<T2>::min(); \
        if (botT1 < botT2) {            \
          if constexpr (!std::is_same_v<signed long long, T2>)                              \
            EXPECT_FALSE(utils::isTypeConversionSafe((T1)(botT2 - 1), (T2)(botT2 - 1)));}\
      }\
    }                                   \
  }                                    \
  if constexpr (!std::is_integral_v<T1> && !std::is_integral_v<T2>) {\
    if constexpr (double(std::numeric_limits<T1>::max()) > double(std::numeric_limits<T2>::max())) {  \
    EXPECT_FALSE(utils::isTypeConversionSafe(((T1)(T1)std::numeric_limits<T2>::max() * (T1)2), (T2)(std::numeric_limits<T2>::max() * (T1)2)));} \
      const double botT1 = double(-(std::numeric_limits<T1>::max())); \
      const double botT2 = double(-(std::numeric_limits<T2>::max())); \
      if constexpr (!std::is_same_v<double, T2>)                              \
        if (botT1 < botT2) {      \
          EXPECT_FALSE(utils::isTypeConversionSafe((T1)(botT2 * 2), (T2)(botT2 * 2)));}                         \
   }
  //} else if constexpr (std::is_integral_v<T1> && !std::is_integral_v<T2>) {\
  //  if constexpr (double(std::numeric_limits<T1>::max()) > double(std::numeric_limits<T2>::max())) {  \
  //  EXPECT_FALSE(format::isTypeConversionSafe(((T1)std::numeric_limits<T2>::max() + (T1)1), std::numeric_limits<T2>::max() + //(T2)1));} \
  //    const intmax_t botT1 = []() {if constexpr (std::is_floating_point_v<T1>) return intmax_t(-(std::numeric_limits<T1>::max())); //else return intmax_t(std::numeric_limits<T1>::min());}(); \
  //    const double botT2 = []() {if constexpr (std::is_floating_point_v<T2>) return double(-(std::numeric_limits<T2>::max())); else //return double(std::numeric_limits<T2>::min());}(); \
  //    if constexpr (!std::is_same_v<double, T2>)                              \
  //      if (double(botT1) < botT2) {      \
  //        EXPECT_FALSE(format::isTypeConversionSafe(botT2 - 1, (T2)(botT2 - 1)));}\
  //} else if constexpr (!std::is_integral_v<T1> && !std::is_integral_v<T2>) {\
  //  if constexpr (double(std::numeric_limits<T1>::max()) > double(std::numeric_limits<T2>::max())) {  \
  //  EXPECT_FALSE(format::isTypeConversionSafe(((T1)std::numeric_limits<T2>::max() + (T1)1), std::numeric_limits<T2>::max() + //(T2)1));} \
  //    const double botT1 = []() {if constexpr (std::is_floating_point_v<T1>) return double(-(std::numeric_limits<T1>::max())); else //return double(std::numeric_limits<T1>::min());}(); \
      const double botT2 = []() {if constexpr (std::is_floating_point_v<T2>) return double(-(std::numeric_limits<T2>::max())); else return double(std::numeric_limits<T2>::min());}(); \
      if constexpr (!std::is_same_v<double, T2>)                              \
        if (double(botT1) < botT2) {      \
          EXPECT_FALSE(format::isTypeConversionSafe(botT2 - 1, (T2)(botT2 - 1)));}\
  }\

#define MATCH_CHECK_ALL_TYPES_INTEGRAL(T1) \
  MATCH_CHECK_ALL_CASES(T1,char) \
  MATCH_CHECK_ALL_CASES(T1,unsigned char) \
  MATCH_CHECK_ALL_CASES(T1,short) \
  MATCH_CHECK_ALL_CASES(T1,unsigned short) \
  MATCH_CHECK_ALL_CASES(T1,int) \
  MATCH_CHECK_ALL_CASES(T1,unsigned int) \
  MATCH_CHECK_ALL_CASES(T1,long long) \
  MATCH_CHECK_ALL_CASES(T1,unsigned long long) \

#define MATCH_CHECK_ALL_TYPES_FLOATING_POINT(T1) \
  MATCH_CHECK_ALL_CASES(T1,float) \
  MATCH_CHECK_ALL_CASES(T1,double)

TEST(isTypeConversionSafe, TypeConversions){
  // int to int
  MATCH_CHECK_ALL_TYPES_INTEGRAL(char);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(unsigned char);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(short);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(unsigned short);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(int);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(unsigned int);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(long long);
  MATCH_CHECK_ALL_TYPES_INTEGRAL(unsigned long long);
  MATCH_CHECK_ALL_TYPES_FLOATING_POINT(float);
  MATCH_CHECK_ALL_TYPES_FLOATING_POINT(double);
  // int to float

  // float to int

  // float to float

}

template <template <typename, typename, typename> typename CSRType1, template <typename , typename , typename > typename CSRType2, typename I1, typename N1, typename V1, typename I2, typename N2, typename V2>
void check_move_csrs(CSRType1<I1, N1, V1>* csr1, CSRType2<I2, N2, V2>* csr2){
  auto nnz1 = csr1->get_num_nnz();
  auto nnz2 = csr2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csr1->get_dimensions()[0],csr2->get_dimensions()[0]);
  ASSERT_EQ(csr1->get_dimensions()[1],csr2->get_dimensions()[1]);
  if constexpr (std::is_same_v<I1, I2> == true)
    EXPECT_EQ((void*)csr1->get_col(), (void*)csr2->get_col());
  else
    EXPECT_NE((void*)csr1->get_col(), (void*)csr2->get_col());
  if constexpr (std::is_same_v<N1, N2> == true)
    EXPECT_EQ((void*)csr1->get_row_ptr(), (void*)csr2->get_row_ptr());
  else
    EXPECT_NE((void*)csr1->get_row_ptr(), (void*)csr2->get_row_ptr());
  if constexpr (std::is_same_v<V1, V2> == true)
    EXPECT_EQ((void*)csr1->get_vals(), (void*)csr2->get_vals());
  else
    EXPECT_NE((void*)csr1->get_vals(), (void*)csr2->get_vals());
}

#define CONVERT_AND_COMPARE_CSR(input_csr, IDT, NNZT, VALT) {auto output_csr =(input_csr)->Convert<format::CSR, IDT, NNZT, VALT>(false) ;  compare_csrs((input_csr), output_csr); delete output_csr;}

#define MOVE_CONVERT_AND_COMPARE_CSR(input_csr, IDT, NNZT, VALT) {auto copy = dynamic_cast<decltype(input_csr)>((input_csr)->Clone()); auto output_csr =(copy)->Convert<format::CSR, IDT, NNZT, VALT>(true) ;  compare_csrs((input_csr), output_csr); check_move_csrs(copy, output_csr); delete output_csr; delete copy;}

#define CONVERT_THEN_COMPARE_CSR(input, operand_csr, IDT, NNZT, VALT) {auto output_csr =(input)->Convert<format::CSR, IDT, NNZT, VALT>(false) ;  compare_csrs((operand_csr), output_csr); delete output_csr;}


template <template <typename, typename, typename> typename COOType1, template <typename , typename , typename > typename COOType2, typename I1, typename N1, typename V1, typename I2, typename N2, typename V2>
void check_move_coos(COOType1<I1, N1, V1>* coo1, COOType2<I2, N2, V2>* coo2){
  auto nnz1 = coo1->get_num_nnz();
  auto nnz2 = coo2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(coo1->get_dimensions()[0],coo2->get_dimensions()[0]);
  ASSERT_EQ(coo1->get_dimensions()[1],coo2->get_dimensions()[1]);
  if constexpr (std::is_same_v<I1, I2> == true)
    EXPECT_EQ((void*)coo1->get_col(), (void*)coo2->get_col());
  else
    EXPECT_NE((void*)coo1->get_col(), (void*)coo2->get_col());
  if constexpr (std::is_same_v<I1, I2> == true)
    EXPECT_EQ((void*)coo1->get_row(), (void*)coo2->get_row());
  else
    EXPECT_NE((void*)coo1->get_row(), (void*)coo2->get_row());
  if constexpr (std::is_same_v<V1, V2> == true)
    EXPECT_EQ((void*)coo1->get_vals(), (void*)coo2->get_vals());
  else
    EXPECT_NE((void*)coo1->get_vals(), (void*)coo2->get_vals());
}

#define CONVERT_AND_COMPARE_COO(input_coo, IDT, NNZT, VALT) {auto output_coo =(input_coo)->Convert<format::COO, IDT, NNZT, VALT>(false) ;  compare_coos((input_coo), output_coo); delete output_coo;}

#define MOVE_CONVERT_AND_COMPARE_COO(input_coo, IDT, NNZT, VALT) {auto copy = dynamic_cast<decltype(input_coo)>((input_coo)->Clone()); auto output_coo =(copy)->Convert<format::COO, IDT, NNZT, VALT>(true) ;  compare_coos((input_coo), output_coo); check_move_coos(copy, output_coo); delete output_coo; delete copy;}

#define CONVERT_THEN_COMPARE_COO(input, operand_coo, IDT, NNZT, VALT) {auto output_coo =(input)->Convert<format::COO, IDT, NNZT, VALT>(false) ;  compare_coos((operand_coo), output_coo); delete output_coo;}

template <template <typename, typename, typename> typename CSCType1, template <typename , typename , typename > typename CSCType2, typename I1, typename N1, typename V1, typename I2, typename N2, typename V2>
void check_move_cscs(CSCType1<I1, N1, V1>* csc1, CSCType2<I2, N2, V2>* csc2){
  auto nnz1 = csc1->get_num_nnz();
  auto nnz2 = csc2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csc1->get_dimensions()[0],csc2->get_dimensions()[0]);
  ASSERT_EQ(csc1->get_dimensions()[1],csc2->get_dimensions()[1]);
  if constexpr (std::is_same_v<N1, N2> == true)
    EXPECT_EQ((void*)csc1->get_col_ptr(), (void*)csc2->get_col_ptr());
  else
    EXPECT_NE((void*)csc1->get_col_ptr(), (void*)csc2->get_col_ptr());
  if constexpr (std::is_same_v<I1, I2> == true)
    EXPECT_EQ((void*)csc1->get_row(), (void*)csc2->get_row());
  else
    EXPECT_NE((void*)csc1->get_row(), (void*)csc2->get_row());
  if constexpr (std::is_same_v<V1, V2> == true)
    EXPECT_EQ((void*)csc1->get_vals(), (void*)csc2->get_vals());
  else
    EXPECT_NE((void*)csc1->get_vals(), (void*)csc2->get_vals());
}

#define CONVERT_AND_COMPARE_CSC(input_csc, IDT, NNZT, VALT) {auto output_csc =(input_csc)->Convert<format::CSC, IDT, NNZT, VALT>(false) ;  compare_cscs((input_csc), output_csc); delete output_csc;}

#define MOVE_CONVERT_AND_COMPARE_CSC(input_csc, IDT, NNZT, VALT) {auto copy = dynamic_cast<decltype(input_csc)>((input_csc)->Clone()); auto output_csc =(copy)->Convert<format::CSC, IDT, NNZT, VALT>(true) ;  compare_cscs((input_csc), output_csc); check_move_cscs(copy, output_csc); delete output_csc; delete copy;}

#define CONVERT_THEN_COMPARE_CSC(input, operand_csc, IDT, NNZT, VALT) {auto output_csc =(input)->Convert<format::CSC, IDT, NNZT, VALT>(false) ;  compare_cscs((operand_csc), output_csc); delete output_csc;}

TEST(FormatOrderTwoTypeConversion, CSR){
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_CSR(&csr, unsigned int, unsigned int, int);
  CONVERT_AND_COMPARE_CSR(&csr, int, unsigned int, int);
  CONVERT_AND_COMPARE_CSR(&csr, unsigned int, int, int);
  CONVERT_AND_COMPARE_CSR(&csr, unsigned int, unsigned int, unsigned int);
  CONVERT_AND_COMPARE_CSR(&csr, int, unsigned int, unsigned int);
  CONVERT_AND_COMPARE_CSR(&csr, unsigned int, int, unsigned int);
  float csr_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double csr_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::CSR<int, int, float> csr_f(
      n, m, csr_row_ptr, csr_col, csr_vals_f, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_CSR(&csr_f, unsigned int, unsigned int, double);
  //sparsebase::format::CSR<int, int, double> csr_d(
  //    n, m, csr_row_ptr, csr_col, csr_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_CSR(&csr_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeMoveConversion, CSR){
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_CSR(&csr, unsigned int, unsigned int, int);
  MOVE_CONVERT_AND_COMPARE_CSR(&csr, int, unsigned int, int);
  MOVE_CONVERT_AND_COMPARE_CSR(&csr, unsigned int, int, int);
  MOVE_CONVERT_AND_COMPARE_CSR(&csr, unsigned int, unsigned int, unsigned int);
  MOVE_CONVERT_AND_COMPARE_CSR(&csr, int, unsigned int, unsigned int);
  MOVE_CONVERT_AND_COMPARE_CSR(&csr, unsigned int, int, unsigned int);
  float csr_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double csr_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::CSR<int, int, float> csr_f(
      n, m, csr_row_ptr, csr_col, csr_vals_f, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_CSR(&csr_f, unsigned int, unsigned int, double);
  //sparsebase::format::CSR<int, int, double> csr_d(
  //    n, m, csr_row_ptr, csr_col, csr_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_CSR(&csr_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeConversion, CSC){
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_CSC(&csc, unsigned int, unsigned int, int);
  CONVERT_AND_COMPARE_CSC(&csc, int, unsigned int, int);
  CONVERT_AND_COMPARE_CSC(&csc, unsigned int, int, int);
  CONVERT_AND_COMPARE_CSC(&csc, unsigned int, unsigned int, unsigned int);
  CONVERT_AND_COMPARE_CSC(&csc, int, unsigned int, unsigned int);
  CONVERT_AND_COMPARE_CSC(&csc, unsigned int, int, unsigned int);
  float csc_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double csc_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::CSC<int, int, float> csc_f(
      n, m, csc_col_ptr, csc_row, csc_vals_f, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_CSC(&csc_f, unsigned int, unsigned int, double);
  //sparsebase::format::CSC<int, int, double> csc_d(
  //    n, m, csc_col_ptr, csc_row, csc_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_CSC(&csc_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeMoveConversion, CSC){
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc, unsigned int, unsigned int, int);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc, int, unsigned int, int);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc, unsigned int, int, int);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc, unsigned int, unsigned int, unsigned int);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc, int, unsigned int, unsigned int);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc, unsigned int, int, unsigned int);
  float csc_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double csc_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::CSC<int, int, float> csc_f(
      n, m, csc_col_ptr, csc_row, csc_vals_f, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_CSC(&csc_f, unsigned int, unsigned int, double);
  //sparsebase::format::CSC<int, int, double> csc_d(
  //    n, m, csc_col_ptr, csc_row, csc_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_CSC(&csc_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeConversion, COO){
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_COO(&coo, unsigned int, unsigned int, int);
  CONVERT_AND_COMPARE_COO(&coo, int, unsigned int, int);
  CONVERT_AND_COMPARE_COO(&coo, unsigned int, int, int);
  CONVERT_AND_COMPARE_COO(&coo, unsigned int, unsigned int, unsigned int);
  CONVERT_AND_COMPARE_COO(&coo, int, unsigned int, unsigned int);
  CONVERT_AND_COMPARE_COO(&coo, unsigned int, int, unsigned int);
  float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double coo_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::COO<int, int, float> coo_f(
      n, m, nnz, coo_row, coo_col, coo_vals_f, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_COO(&coo_f, unsigned int, unsigned int, double);
  //sparsebase::format::COO<int, int, double> coo_d(
  //    n, m, nnz, coo_row, coo_col, coo_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_COO(&coo_d, unsigned int, unsigned int, float);
}

TEST(FormatOrderTwoTypeMoveConversion, COO){
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_COO(&coo, unsigned int, unsigned int, int);
  MOVE_CONVERT_AND_COMPARE_COO(&coo, int, unsigned int, int);
  MOVE_CONVERT_AND_COMPARE_COO(&coo, unsigned int, int, int);
  MOVE_CONVERT_AND_COMPARE_COO(&coo, unsigned int, unsigned int, unsigned int);
  MOVE_CONVERT_AND_COMPARE_COO(&coo, int, unsigned int, unsigned int);
  MOVE_CONVERT_AND_COMPARE_COO(&coo, unsigned int, int, unsigned int);
  float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double coo_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::COO<int, int, float> coo_f(
      n, m, nnz, coo_row, coo_col, coo_vals_f, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_COO(&coo_f, unsigned int, unsigned int, double);
  //sparsebase::format::COO<int, int, double> coo_d(
  //    n, m, nnz, coo_row, coo_col, coo_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_COO(&coo_d, unsigned int, unsigned int, float);
}

TEST(FormatOrderTwoFormatAndTypeConversion, CSR){
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);

  auto csc_uui = csc.Convert<format::CSC, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_CSC(&csr, csc_uui, unsigned int, unsigned int, int);

  auto coo_uui = coo.Convert<format::COO, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_COO(&csr, coo_uui, unsigned int, unsigned int, int);

  float csr_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::CSR<int, int, float> csr_f(
      n, m, csr_row_ptr, csr_col, csr_vals_f, sparsebase::format::kNotOwned);
  auto csc_f = csr_f.Convert<format::CSC>();
  auto coo_f = csr_f.Convert<format::COO>();

  CONVERT_THEN_COMPARE_CSC(&csr_f, csc_f, unsigned int, unsigned int, double);
  CONVERT_THEN_COMPARE_COO(&csr_f, coo_f, unsigned int, unsigned int, double);
}

TEST(FormatOrderTwoFormatAndTypeConversion, COO){
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);

  auto csc_uui = coo.Convert<format::CSC, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_CSC(&coo, csc_uui, unsigned int, unsigned int, int);

  auto csr_uui = coo.Convert<format::CSR, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_CSR(&coo, csr_uui, unsigned int, unsigned int, int);

  float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::COO<int, int, float> coo_f(
      n, m, nnz, coo_row, coo_col, coo_vals_f, sparsebase::format::kNotOwned);
  auto csc_f = coo_f.Convert<format::CSC>();
  auto csr_f = coo_f.Convert<format::CSR>();

  CONVERT_THEN_COMPARE_CSC(&coo_f, csc_f, unsigned int, unsigned int, double);
  CONVERT_THEN_COMPARE_CSR(&coo_f, csr_f, unsigned int, unsigned int, double);
}
//TEST(FormatOrderTwoFormatAndTypeConversion, CSC){
//  sparsebase::format::CSR<int, int, int> csr(
//      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
//  sparsebase::format::COO<int, int, int> coo(
//      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
//  sparsebase::format::CSC<int, int, int> csc(
//      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
//
//  // need multi-step conversion
//  //auto csr_uui = csc.Convert<format::CSR, unsigned int, unsigned int, int>(false);
//  //CONVERT_THEN_COMPARE_CSR(&csc, csr_uui, unsigned int, unsigned int, int);
//
//  auto coo_uui = csc.Convert<format::COO, unsigned int, unsigned int, int>(false);
//  CONVERT_THEN_COMPARE_COO(&csc, coo_uui, unsigned int, unsigned int, int);
//
//  float csc_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
//  sparsebase::format::CSC<int, int, float> csc_f(
//      n, m, csc_col_ptr, csc_row, csc_vals_f, sparsebase::format::kNotOwned);
//  //auto csr_f = csc_f.Convert<format::CSR>(); // need to implement multi-step conversion!
//  auto coo_f = csc_f.Convert<format::COO>();
//
//  //CONVERT_THEN_COMPARE_CSR(&csc_f, csr_f, unsigned int, unsigned int, double);
//  CONVERT_THEN_COMPARE_COO(&csc_f, coo_f, unsigned int, unsigned int, double);
//}

template <typename ArrayType1, typename ArrayType2>
void compare_arrays(ArrayType1* array1, ArrayType2* array2){
  auto nnz1 = array1->get_num_nnz();
  auto nnz2 = array2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(array1->get_dimensions()[0],array2->get_dimensions()[0]);
  compare_arrays(array1->get_vals(),array2->get_vals(), nnz1, "vals");
}

template <template <typename> typename ArrayType1, template <typename> typename ArrayType2, typename V1, typename V2>
void check_move_arrays(ArrayType1<V1>* array1, ArrayType2<V2>* array2){
  auto nnz1 = array1->get_num_nnz();
  auto nnz2 = array2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(array1->get_dimensions()[0],array2->get_dimensions()[0]);
  if constexpr (std::is_same_v<V1, V2>)
    EXPECT_EQ((void*)array1->get_vals(), (void*)array2->get_vals());
  else
    EXPECT_NE((void*)array1->get_vals(), (void*)array2->get_vals());
}

#define CONVERT_AND_COMPARE_Array(input_array, VALT) {auto output_array =(input_array)->Convert<format::Array, VALT>(false) ;  compare_arrays((input_array), output_array); delete output_array;}

#define MOVE_CONVERT_AND_COMPARE_Array(input_arr, VALT) {auto copy = dynamic_cast<decltype(input_arr)>((input_arr)->Clone()); auto output_arr =(copy)->Convert<format::Array, VALT>(true) ;  compare_arrays((input_arr), output_arr); check_move_arrays(copy, output_arr); delete output_arr; delete copy;}

TEST(FormatOrderOneTypeConversion, Array){
  sparsebase::format::Array<int> array(
      nnz, coo_vals, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_Array(&array, unsigned int);
  float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double coo_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::Array<float> array_f(
      nnz, coo_vals_f, sparsebase::format::kNotOwned);
  CONVERT_AND_COMPARE_Array(&array_f,double);
  //sparsebase::format::Array<double> array_d(
  //    nnz, coo_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_Array(&array_d,float);
}
TEST(FormatOrderOneTypeMoveConversion, Array){
  sparsebase::format::Array<int> array(
      nnz, coo_vals, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_Array(&array, unsigned int);
  float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  double coo_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::Array<float> array_f(
      nnz, coo_vals_f, sparsebase::format::kNotOwned);
  MOVE_CONVERT_AND_COMPARE_Array(&array_f,double);
  //sparsebase::format::Array<double> array_d(
  //    nnz, coo_vals_d, sparsebase::format::kNotOwned);
  //CONVERT_AND_COMPARE_Array(&array_d,float);
}

format::Format* returnCoo(format::Format*, context::Context*){
  return new format::COO<int, int, int>(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
}

format::Format* returnCsr(format::Format*, context::Context*){
  return new format::CSR<int, int, int>
      (
          n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned)
      ;
}

format::Format* returnCsc(format::Format*, context::Context*){
  return new format::CSC<int, int, int>
      (
          n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned)
      ;
}

#define CHECK_FIRST_FORMAT(v1, v2, v3) \
EXPECT_EQ((v1->Is<format::CSR<TYPE>>()), true);\
EXPECT_EQ((v2->Is<format::COO<TYPE>>()), true); \
EXPECT_EQ((v3->Is<format::CSC<TYPE>>()), true);

TEST(ApplyConversionSchema, All){
#define ConversionPairVector std::vector<std::tuple<ConversionFunction, context::Context*>>
#define TYPE int, int, int
  utils::converter::ConversionSchema schema;
  std::vector<std::vector<format::Format*>> output;
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
  context::CPUContext cpu;
  // don't convert anything
  schema.clear();
  schema.insert(schema.end(), {{}, {}, {}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc});
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 1);
  EXPECT_EQ(output[1].size(), 1);
  EXPECT_EQ(output[1].size(), 1);
  // Convert first once
  schema.clear();
  schema.insert(schema.end(),
                {std::make_tuple(ConversionPairVector{std::make_tuple(returnCoo, &cpu)}, 1), {}, {}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc});
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 2);
  EXPECT_EQ((output[0][1]->Is<format::COO<TYPE>>()), true);
  EXPECT_EQ(output[1].size(), 1);
  EXPECT_EQ(output[2].size(), 1);
  // Convert second and third once
  schema.clear();
  schema.insert(schema.end(),
                {{},
                 {std::make_tuple(ConversionPairVector{std::make_tuple(returnCsr, &cpu)}, 1)},
                 {std::make_tuple(ConversionPairVector{std::make_tuple(returnCsr, &cpu)}, 1)}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc});
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 1);
  EXPECT_EQ(output[1].size(), 2);
  EXPECT_EQ((output[1][1]->Is<format::CSR<TYPE>>()), true);
  delete output[1][1];
  EXPECT_EQ(output[2].size(), 2);
  EXPECT_EQ((output[2][1]->Is<format::CSR<TYPE>>()), true);
  delete output[2][1];
  // Convert second twice and third once
  schema.clear();
  schema.insert(schema.end(),
                {{},
                 {std::make_tuple(ConversionPairVector{std::make_tuple(returnCsr, &cpu), std::make_tuple(returnCsc, &cpu)}, 1)},
                 {std::make_tuple(ConversionPairVector{std::make_tuple(returnCsr, &cpu)}, 1)}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc});
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 1);
  EXPECT_EQ(output[1].size(), 3);
  EXPECT_EQ((output[1][1]->Is<format::CSR<TYPE>>()), true);
  delete output[1][1];
  EXPECT_EQ((output[1][2]->Is<format::CSC<TYPE>>()), true);
  delete output[1][2];
  EXPECT_EQ(output[2].size(), 2);
  EXPECT_EQ((output[2][1]->Is<format::CSR<TYPE>>()), true);
  delete output[2][1];
#undef ConversionPair
#undef TYPE
}

#define ConversionPairVector std::vector<std::tuple<ConditionalConversionFunction, context::Context*>>
#define TYPE int, int, int
class ConversionChainFixture : public ::testing::Test {
protected:
  struct FakeContext : context::ContextImplementation<FakeContext> {
    virtual bool IsEquivalent(Context * i) const {
        return i->get_context_type_member() == this->get_context_type_member();
    };
  };
  void SetUp() override {

    csr = new sparsebase::format::CSR<int, int, int>(
        n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
    coo = new sparsebase::format::COO<int, int, int>(
        n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
    csc = new format::CSC<int, int, int>(
        n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
    c.ClearConversionFunctions(csr->get_format_id(), coo->get_format_id(), false);
    c.ClearConversionFunctions(csr->get_format_id(), csc->get_format_id(), false);
    c.ClearConversionFunctions(coo->get_format_id(), csc->get_format_id(), false);
    c.ClearConversionFunctions(coo->get_format_id(), csr->get_format_id(), false);
    c.ClearConversionFunctions(csc->get_format_id(), csr->get_format_id(), false);
    c.ClearConversionFunctions(csc->get_format_id(), coo->get_format_id(), false);
    c.RegisterConversionFunction(
        csr->get_format_id(), coo->get_format_id(), returnCoo,
        [](context::Context *, context::Context *to) -> bool {
          return to->get_context_type_member() ==
                 context::CPUContext::get_context_type();
        });
    c.RegisterConversionFunction(
        coo->get_format_id(), csr->get_format_id(), returnCsr,
        [](context::Context *from, context::Context *to) -> bool {
          return to->get_context_type_member() ==
                 FakeContext::get_context_type();
        });
  }
  void TearDown() override {
    delete csr;
    delete coo;
    delete csc;
  }
  sparsebase::format::CSR<int, int, int>* csr;
  sparsebase::format::COO<int, int, int>* coo;
  sparsebase::format::CSC<int, int, int>* csc;
  ConverterOrderTwo<TYPE> c;
  context::CPUContext cpu_c;
  FakeContext fake_c;
};

TEST_F(ConversionChainFixture, SingleStep){
  utils::converter::ConversionSchema schema;
  std::vector<std::vector<format::Format*>> output;
  ConversionChain chain;
  format::Format* output_format;
  // single format
  // No conversion -- wrong context
  chain = c.GetConversionChain(coo->get_format_id(), &cpu_c, csr->get_format_id(),{&cpu_c}, false);
  EXPECT_EQ(chain.has_value(), false);
  // No conversion -- no function
  chain = c.GetConversionChain(csc->get_format_id(), &cpu_c, csr->get_format_id(),{&cpu_c}, false);
  EXPECT_EQ(chain.has_value(), false);
  // One conversion -- cpu context
  chain = c.GetConversionChain(csr->get_format_id(), &cpu_c, coo->get_format_id(),{&cpu_c}, false);
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()), context::CPUContext::get_context_type());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_format_id(), (format::COO<TYPE>::get_format_id_static()));
  delete output_format;
  // One conversion -- fake context
  chain = c.GetConversionChain(coo->get_format_id(), &cpu_c, csr->get_format_id(),{&fake_c}, false);
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()), FakeContext::get_context_type());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_format_id(), (format::CSR<TYPE>::get_format_id_static()));
  delete output_format;
  // Multiple conversions
}

TEST_F(ConversionChainFixture, MultiStep){
  utils::converter::ConversionSchema schema;
  std::vector<std::vector<format::Format*>> output;
  ConversionChain chain;
  format::Format* output_format;
  c.RegisterConversionFunction(
      coo->get_format_id(), csc->get_format_id(), returnCsc,
      [](context::Context *from, context::Context *to) -> bool {
        return to->get_context_type_member() == FakeContext::get_context_type();
      });
  // single format
  // No conversion -- only one needed context
  chain = c.GetConversionChain(csr->get_format_id(), &cpu_c, csc->get_format_id(),{&cpu_c}, false);
  EXPECT_EQ(chain.has_value(), false);
  // No conversion -- only one needed context
  chain = c.GetConversionChain(csr->get_format_id(), &cpu_c, csc->get_format_id(),{&fake_c}, false);
  EXPECT_EQ(chain.has_value(), false);
  // Two conversions -- only one needed context
  chain = c.GetConversionChain(csr->get_format_id(), &cpu_c, csc->get_format_id(),{&fake_c, &cpu_c}, false);
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 2);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()), context::CPUContext::get_context_type());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_format_id(), (format::COO<TYPE>::get_format_id_static()));
  delete output_format;
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[1])->get_context_type_member()), FakeContext::get_context_type());
  output_format = (std::get<0>(std::get<0>(*chain)[1]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_format_id(), (format::CSC<TYPE>::get_format_id_static()));
  delete output_format;
}
#undef ConversionPair
#undef TYPE

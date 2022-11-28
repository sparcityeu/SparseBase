#include <iostream>

#include "common.inc"
#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
using namespace sparsebase;
using namespace converter;

TEST(Converter, ClearingAllFunctions) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  converterOrderTwo.ClearConversionFunctions();
  EXPECT_THROW(
      (converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(
          &csr, &cpu_context)),
      sparsebase::utils::ConversionException);
  converterOrderTwo.RegisterConversionFunction(
      sparsebase::format::CSR<int, int, int>::get_id_static(),
      sparsebase::format::COO<int, int, int>::get_id_static(),
      [](sparsebase::format::Format*, sparsebase::context::Context*)
          -> sparsebase::format::Format* { return nullptr; },
      [](sparsebase::context::Context*, sparsebase::context::Context*) -> bool {
        return true;
      });
  EXPECT_EQ((converterOrderTwo.Convert(
                &csr, sparsebase::format::COO<int, int, int>::get_id_static(),
                &cpu_context)),
            nullptr);
}

TEST(Converter, ClearingASingleDirection) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::converter::ConverterOrderTwo<int, int, int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;

  converterOrderTwo.ClearConversionFunctions(
      sparsebase::format::COO<int, int, int>::get_id_static(),
      sparsebase::format::CSR<int, int, int>::get_id_static());
  EXPECT_NO_THROW(
      (converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(
          &csr, &cpu_context)));
  converterOrderTwo.ClearConversionFunctions(
      sparsebase::format::CSR<int, int, int>::get_id_static(),
      sparsebase::format::COO<int, int, int>::get_id_static());
  EXPECT_THROW(
      (converterOrderTwo.Convert<sparsebase::format::COO<int, int, int>>(
          &csr, &cpu_context)),
      sparsebase::utils::ConversionException);
  converterOrderTwo.RegisterConversionFunction(
      sparsebase::format::CSR<int, int, int>::get_id_static(),
      sparsebase::format::COO<int, int, int>::get_id_static(),
      [](sparsebase::format::Format*, sparsebase::context::Context*)
          -> sparsebase::format::Format* { return nullptr; },
      [](sparsebase::context::Context*, sparsebase::context::Context*) -> bool {
        return true;
      });
  EXPECT_EQ((converterOrderTwo.Convert(
                &csr, sparsebase::format::COO<int, int, int>::get_id_static(),
                &cpu_context)),
            nullptr);
}

#define MATCH_CHECKING_ONEVAL(T1, T2, val)                                \
  EXPECT_TRUE(utils::isTypeConversionSafe((T1)(val), (T2)(val)));         \
  EXPECT_TRUE(utils::isTypeConversionSafe((T2)(val), (T1)(val)));         \
  if ((val) != 0) {                                                       \
    EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-(val)), (T2)(val)));   \
    EXPECT_FALSE(utils::isTypeConversionSafe((T1)((val)), (T2)(-(val)))); \
    EXPECT_FALSE(utils::isTypeConversionSafe((T2)(-(val)), (T1)(val)));   \
    EXPECT_FALSE(utils::isTypeConversionSafe((T2)((val)), (T1)(-(val)))); \
  }
#define MATCH_CHECKING_SIGNED_MISMATCH(T1, T2)                    \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(-1), (T1)(-1))); \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(-1.5), (T1)(-1.5)));

#define MATCH_CHECKING_SIGNED_MATCH(T1, T2)                       \
  EXPECT_TRUE(format::isTypeConversionSafe((T2)(-1), (T1)(-1)));  \
  if constexpr (std::is_integral_v<T1> == std::is_integral_v<T2>) \
    EXPECT_TRUE(format::isTypeConversionSafe((T2)(-2.5), (T1)(-2.5)));

#define MATCH_CHECKING_INT_FLOAT_MISMATCH(T1, T2)                     \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(-3.5), (T1)(-3.5))); \
  EXPECT_FALSE(format::isTypeConversionSafe((T2)(4.5), (T1)(4.5)));

#define MATCH_CHECKING_INT_FLOAT_MATCH(T1, T2)                       \
  EXPECT_TRUE(format::isTypeConversionSafe((T2)(-5.5), (T1)(-5.5))); \
  EXPECT_TRUE(format::isTypeConversionSafe((T2)(6.5), (T1)(6.5)));

#include <limits>
#define MATCH_CHECK_ALL_CASES(T1, T2)                                       \
  MATCH_CHECKING_ONEVAL(T1, T2, 0)                                          \
  MATCH_CHECKING_ONEVAL(T1, T2, 1)                                          \
  if constexpr (std::is_integral_v<T1>) {                                   \
    if constexpr (std::is_signed_v<T1>) {                                   \
      if constexpr (std::is_integral_v<T2>) {                               \
        if constexpr (std::is_signed_v<T2>)                                 \
          EXPECT_TRUE(utils::isTypeConversionSafe((T1)(-1), (T2)(-1)));     \
        else                                                                \
          EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-1), (T2)(-1)));    \
      } else {                                                              \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(2), (T2)(2)));        \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-2), (T2)(-2)));      \
      }                                                                     \
    } else {                                                                \
    }                                                                       \
  } else {                                                                  \
    if constexpr (std::is_integral_v<T2>) {                                 \
      EXPECT_FALSE(utils::isTypeConversionSafe((T1)(1.5), (T2)(1.5)));      \
      EXPECT_FALSE(utils::isTypeConversionSafe((T1)(3), (T2)(3)));          \
      if constexpr (std::is_signed_v<T2>) {                                 \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-1.5), (T2)(-1.5)));  \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-3), (T2)(-3)));      \
      } else {                                                              \
        EXPECT_FALSE(utils::isTypeConversionSafe((T1)(-3), (T2)(-3)));      \
      }                                                                     \
    }                                                                       \
  }                                                                         \
  if constexpr (std::is_floating_point_v<T2> == false) {                    \
    if constexpr (std::is_integral_v<T1> == true &&                         \
                  std::is_integral_v<T2> == true) {                         \
      if constexpr (std::is_floating_point_v<T1> == false &&                \
                    std::uintmax_t(std::numeric_limits<T1>::max()) >        \
                        std::uintmax_t(std::numeric_limits<T2>::max())) {   \
        EXPECT_FALSE(utils::isTypeConversionSafe(                           \
            (T1)(std::numeric_limits<T2>::max() + (unsigned long long)1),   \
            (T2)((unsigned long long)std::numeric_limits<T2>::max() +       \
                 (unsigned long long)1)));                                  \
      }                                                                     \
      {                                                                     \
        const intmax_t botT1 = std::numeric_limits<T1>::min();              \
        const intmax_t botT2 = std::numeric_limits<T2>::min();              \
        if (botT1 < botT2) {                                                \
          if constexpr (!std::is_same_v<signed long long, T2>)              \
            EXPECT_FALSE(utils::isTypeConversionSafe((T1)(botT2 - 1),       \
                                                     (T2)(botT2 - 1)));     \
        }                                                                   \
      }                                                                     \
    }                                                                       \
  }                                                                         \
  if constexpr (!std::is_integral_v<T1> && !std::is_integral_v<T2>) {       \
    if constexpr (double(std::numeric_limits<T1>::max()) >                  \
                  double(std::numeric_limits<T2>::max())) {                 \
      EXPECT_FALSE(utils::isTypeConversionSafe(                             \
          ((T1)(T1)std::numeric_limits<T2>::max() * (T1)2),                 \
          (T2)(std::numeric_limits<T2>::max() * (T1)2)));                   \
    }                                                                       \
    const double botT1 = double(-(std::numeric_limits<T1>::max()));         \
    const double botT2 = double(-(std::numeric_limits<T2>::max()));         \
    if constexpr (!std::is_same_v<double, T2>)                              \
      if (botT1 < botT2) {                                                  \
        EXPECT_FALSE(                                                       \
            utils::isTypeConversionSafe((T1)(botT2 * 2), (T2)(botT2 * 2))); \
      }                                                                     \
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

#define MATCH_CHECK_ALL_TYPES_INTEGRAL(T1)  \
  MATCH_CHECK_ALL_CASES(T1, char)           \
  MATCH_CHECK_ALL_CASES(T1, unsigned char)  \
  MATCH_CHECK_ALL_CASES(T1, short)          \
  MATCH_CHECK_ALL_CASES(T1, unsigned short) \
  MATCH_CHECK_ALL_CASES(T1, int)            \
  MATCH_CHECK_ALL_CASES(T1, unsigned int)   \
  MATCH_CHECK_ALL_CASES(T1, long long)      \
  MATCH_CHECK_ALL_CASES(T1, unsigned long long)

#define MATCH_CHECK_ALL_TYPES_FLOATING_POINT(T1) \
  MATCH_CHECK_ALL_CASES(T1, float)               \
  MATCH_CHECK_ALL_CASES(T1, double)

TEST(isTypeConversionSafe, TypeConversions) {
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

template <template <typename, typename, typename> typename CSRType1,
          template <typename, typename, typename> typename CSRType2,
          typename I1, typename N1, typename V1, typename I2, typename N2,
          typename V2>
void check_move_csrs(CSRType1<I1, N1, V1>* csr1, CSRType2<I2, N2, V2>* csr2) {
  auto nnz1 = csr1->get_num_nnz();
  auto nnz2 = csr2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csr1->get_dimensions()[0], csr2->get_dimensions()[0]);
  ASSERT_EQ(csr1->get_dimensions()[1], csr2->get_dimensions()[1]);
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

#define CONVERT_AND_COMPARE_CSR(input_csr, IDT, NNZT, VALT)        \
  {                                                                \
    auto output_csr =                                              \
        (input_csr)->Convert<format::CSR, IDT, NNZT, VALT>(false); \
    compare_csrs((input_csr), output_csr);                         \
    delete output_csr;                                             \
  }

#define MOVE_CONVERT_AND_COMPARE_CSR(input_csr, IDT, NNZT, VALT)           \
  {                                                                        \
    auto copy = dynamic_cast<decltype(input_csr)>((input_csr)->Clone());   \
    auto output_csr = (copy)->Convert<format::CSR, IDT, NNZT, VALT>(true); \
    compare_csrs((input_csr), output_csr);                                 \
    check_move_csrs(copy, output_csr);                                     \
    delete output_csr;                                                     \
    delete copy;                                                           \
  }

#define CONVERT_THEN_COMPARE_CSR(input, operand_csr, IDT, NNZT, VALT)        \
  {                                                                          \
    auto output_csr = (input)->Convert<format::CSR, IDT, NNZT, VALT>(false); \
    compare_csrs((operand_csr), output_csr);                                 \
    delete output_csr;                                                       \
  }

template <template <typename, typename, typename> typename COOType1,
          template <typename, typename, typename> typename COOType2,
          typename I1, typename N1, typename V1, typename I2, typename N2,
          typename V2>
void check_move_coos(COOType1<I1, N1, V1>* coo1, COOType2<I2, N2, V2>* coo2) {
  auto nnz1 = coo1->get_num_nnz();
  auto nnz2 = coo2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(coo1->get_dimensions()[0], coo2->get_dimensions()[0]);
  ASSERT_EQ(coo1->get_dimensions()[1], coo2->get_dimensions()[1]);
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

#define CONVERT_AND_COMPARE_COO(input_coo, IDT, NNZT, VALT)        \
  {                                                                \
    auto output_coo =                                              \
        (input_coo)->Convert<format::COO, IDT, NNZT, VALT>(false); \
    compare_coos((input_coo), output_coo);                         \
    delete output_coo;                                             \
  }

#define MOVE_CONVERT_AND_COMPARE_COO(input_coo, IDT, NNZT, VALT)           \
  {                                                                        \
    auto copy = dynamic_cast<decltype(input_coo)>((input_coo)->Clone());   \
    auto output_coo = (copy)->Convert<format::COO, IDT, NNZT, VALT>(true); \
    compare_coos((input_coo), output_coo);                                 \
    check_move_coos(copy, output_coo);                                     \
    delete output_coo;                                                     \
    delete copy;                                                           \
  }

#define CONVERT_THEN_COMPARE_COO(input, operand_coo, IDT, NNZT, VALT)        \
  {                                                                          \
    auto output_coo = (input)->Convert<format::COO, IDT, NNZT, VALT>(false); \
    compare_coos((operand_coo), output_coo);                                 \
    delete output_coo;                                                       \
  }

template <template <typename, typename, typename> typename CSCType1,
          template <typename, typename, typename> typename CSCType2,
          typename I1, typename N1, typename V1, typename I2, typename N2,
          typename V2>
void check_move_cscs(CSCType1<I1, N1, V1>* csc1, CSCType2<I2, N2, V2>* csc2) {
  auto nnz1 = csc1->get_num_nnz();
  auto nnz2 = csc2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(csc1->get_dimensions()[0], csc2->get_dimensions()[0]);
  ASSERT_EQ(csc1->get_dimensions()[1], csc2->get_dimensions()[1]);
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

#define CONVERT_AND_COMPARE_CSC(input_csc, IDT, NNZT, VALT)        \
  {                                                                \
    auto output_csc =                                              \
        (input_csc)->Convert<format::CSC, IDT, NNZT, VALT>(false); \
    compare_cscs((input_csc), output_csc);                         \
    delete output_csc;                                             \
  }

#define MOVE_CONVERT_AND_COMPARE_CSC(input_csc, IDT, NNZT, VALT)           \
  {                                                                        \
    auto copy = dynamic_cast<decltype(input_csc)>((input_csc)->Clone());   \
    auto output_csc = (copy)->Convert<format::CSC, IDT, NNZT, VALT>(true); \
    compare_cscs((input_csc), output_csc);                                 \
    check_move_cscs(copy, output_csc);                                     \
    delete output_csc;                                                     \
    delete copy;                                                           \
  }

#define CONVERT_THEN_COMPARE_CSC(input, operand_csc, IDT, NNZT, VALT)        \
  {                                                                          \
    auto output_csc = (input)->Convert<format::CSC, IDT, NNZT, VALT>(false); \
    compare_cscs((operand_csc), output_csc);                                 \
    delete output_csc;                                                       \
  }

TEST(FormatOrderTwoTypeConversion, CSR) {
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
  // sparsebase::format::CSR<int, int, double> csr_d(
  //    n, m, csr_row_ptr, csr_col, csr_vals_d, sparsebase::format::kNotOwned);
  // CONVERT_AND_COMPARE_CSR(&csr_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeMoveConversion, CSR) {
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
  // sparsebase::format::CSR<int, int, double> csr_d(
  //    n, m, csr_row_ptr, csr_col, csr_vals_d, sparsebase::format::kNotOwned);
  // CONVERT_AND_COMPARE_CSR(&csr_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeConversion, CSC) {
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
  // sparsebase::format::CSC<int, int, double> csc_d(
  //    n, m, csc_col_ptr, csc_row, csc_vals_d, sparsebase::format::kNotOwned);
  // CONVERT_AND_COMPARE_CSC(&csc_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeMoveConversion, CSC) {
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
  // sparsebase::format::CSC<int, int, double> csc_d(
  //    n, m, csc_col_ptr, csc_row, csc_vals_d, sparsebase::format::kNotOwned);
  // CONVERT_AND_COMPARE_CSC(&csc_d, unsigned int, unsigned int, float);
}
TEST(FormatOrderTwoTypeConversion, COO) {
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
  // sparsebase::format::COO<int, int, double> coo_d(
  //    n, m, nnz, coo_row, coo_col, coo_vals_d, sparsebase::format::kNotOwned);
  // CONVERT_AND_COMPARE_COO(&coo_d, unsigned int, unsigned int, float);
}

TEST(FormatOrderTwoTypeMoveConversion, COO) {
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
  // sparsebase::format::COO<int, int, double> coo_d(
  //    n, m, nnz, coo_row, coo_col, coo_vals_d, sparsebase::format::kNotOwned);
  // CONVERT_AND_COMPARE_COO(&coo_d, unsigned int, unsigned int, float);
}

TEST(FormatOrderTwoFormatAndTypeConversion, CSR) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);

  auto csc_uui =
      csc.Convert<format::CSC, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_CSC(&csr, csc_uui, unsigned int, unsigned int, int);

  auto coo_uui =
      coo.Convert<format::COO, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_COO(&csr, coo_uui, unsigned int, unsigned int, int);

  float csr_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::CSR<int, int, float> csr_f(
      n, m, csr_row_ptr, csr_col, csr_vals_f, sparsebase::format::kNotOwned);
  auto csc_f = csr_f.Convert<format::CSC>();
  auto coo_f = csr_f.Convert<format::COO>();

  CONVERT_THEN_COMPARE_CSC(&csr_f, csc_f, unsigned int, unsigned int, double);
  CONVERT_THEN_COMPARE_COO(&csr_f, coo_f, unsigned int, unsigned int, double);
}

TEST(FormatOrderTwoFormatAndTypeConversion, COO) {
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::format::COO<int, int, int> coo(
      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::format::CSC<int, int, int> csc(
      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);

  auto csc_uui =
      coo.Convert<format::CSC, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_CSC(&coo, csc_uui, unsigned int, unsigned int, int);

  auto csr_uui =
      coo.Convert<format::CSR, unsigned int, unsigned int, int>(false);
  CONVERT_THEN_COMPARE_CSR(&coo, csr_uui, unsigned int, unsigned int, int);

  float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
  sparsebase::format::COO<int, int, float> coo_f(
      n, m, nnz, coo_row, coo_col, coo_vals_f, sparsebase::format::kNotOwned);
  auto csc_f = coo_f.Convert<format::CSC>();
  auto csr_f = coo_f.Convert<format::CSR>();

  CONVERT_THEN_COMPARE_CSC(&coo_f, csc_f, unsigned int, unsigned int, double);
  CONVERT_THEN_COMPARE_CSR(&coo_f, csr_f, unsigned int, unsigned int, double);
}
// TEST(FormatOrderTwoFormatAndTypeConversion, CSC){
//  sparsebase::format::CSR<int, int, int> csr(
//      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
//  sparsebase::format::COO<int, int, int> coo(
//      n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
//  sparsebase::format::CSC<int, int, int> csc(
//      n, m, csc_col_ptr, csc_row, csc_vals, sparsebase::format::kNotOwned);
//
//  // need multi-step conversion
//  //auto csr_uui = csc.Convert<format::CSR, unsigned int, unsigned int,
//  int>(false);
//  //CONVERT_THEN_COMPARE_CSR(&csc, csr_uui, unsigned int, unsigned int, int);
//
//  auto coo_uui = csc.Convert<format::COO, unsigned int, unsigned int,
//  int>(false); CONVERT_THEN_COMPARE_COO(&csc, coo_uui, unsigned int, unsigned
//  int, int);
//
//  float csc_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
//  sparsebase::format::CSC<int, int, float> csc_f(
//      n, m, csc_col_ptr, csc_row, csc_vals_f, sparsebase::format::kNotOwned);
//  //auto csr_f = csc_f.Convert<format::CSR>(); // need to implement multi-step
//  conversion! auto coo_f = csc_f.Convert<format::COO>();
//
//  //CONVERT_THEN_COMPARE_CSR(&csc_f, csr_f, unsigned int, unsigned int,
//  double); CONVERT_THEN_COMPARE_COO(&csc_f, coo_f, unsigned int, unsigned int,
//  double);
//}

format::Format* returnCoo(format::Format*, context::Context*) {
  return new format::COO<int, int, int>(n, m, nnz, coo_row, coo_col, coo_vals,
                                        sparsebase::format::kNotOwned);
}

format::Format* returnCsr(format::Format*, context::Context*) {
  return new format::CSR<int, int, int>(n, m, csr_row_ptr, csr_col, csr_vals,
                                        sparsebase::format::kNotOwned);
}

format::Format* returnCsc(format::Format*, context::Context*) {
  return new format::CSC<int, int, int>(n, m, csc_col_ptr, csc_row, csc_vals,
                                        sparsebase::format::kNotOwned);
}

#define CHECK_FIRST_FORMAT(v1, v2, v3)            \
  EXPECT_EQ((v1->Is<format::CSR<TYPE>>()), true); \
  EXPECT_EQ((v2->Is<format::COO<TYPE>>()), true); \
  EXPECT_EQ((v3->Is<format::CSC<TYPE>>()), true);

TEST(ApplyConversionSchema, All) {
#define ConversionPairVector \
  std::vector<               \
      std::tuple<ConversionFunction, context::Context*, utils::CostType>>
#define TYPE int, int, int
  converter::ConversionSchema schema;
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
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, false);
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 1);
  EXPECT_EQ(output[1].size(), 1);
  EXPECT_EQ(output[1].size(), 1);
  // Convert first once
  schema.clear();
  schema.insert(
      schema.end(),
      {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCoo, &cpu, 1)}, 1),
       {},
       {}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, false);
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 2);
  EXPECT_EQ((output[0][1]->Is<format::COO<TYPE>>()), true);
  EXPECT_EQ(output[1].size(), 1);
  EXPECT_EQ(output[2].size(), 1);
  // Convert second and third once
  schema.clear();
  schema.insert(
      schema.end(),
      {{},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1)}, 1)},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1)}, 1)}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, false);
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
  schema.insert(
      schema.end(),
      {{},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1),
                                std::make_tuple(returnCsc, &cpu, 1)},
           1)},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1)}, 1)}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, false);
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

TEST(ApplyConversionSchema, ClearIntermediate) {
#define ConversionPairVector \
  std::vector<               \
      std::tuple<ConversionFunction, context::Context*, utils::CostType>>
#define TYPE int, int, int
  converter::ConversionSchema schema;
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
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, true);
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 1);
  EXPECT_EQ(output[1].size(), 1);
  EXPECT_EQ(output[1].size(), 1);
  // Convert first once
  schema.clear();
  schema.insert(
      schema.end(),
      {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCoo, &cpu, 1)}, 1),
       {},
       {}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, true);
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 2);
  EXPECT_EQ((output[0][1]->Is<format::COO<TYPE>>()), true);
  EXPECT_EQ(output[1].size(), 1);
  EXPECT_EQ(output[2].size(), 1);
  // Convert second and third once
  schema.clear();
  schema.insert(
      schema.end(),
      {{},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1)}, 1)},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1)}, 1)}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, true);
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
  schema.insert(
      schema.end(),
      {{},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1),
                                std::make_tuple(returnCsc, &cpu, 1)},
           1)},
       {std::make_tuple(
           ConversionPairVector{std::make_tuple(returnCsr, &cpu, 1)}, 1)}});
  output = Converter::ApplyConversionSchema(schema, {&csr, &coo, &csc}, true);
  EXPECT_EQ(output.size(), 3);
  CHECK_FIRST_FORMAT(output[0][0], output[1][0], output[2][0]);
  EXPECT_EQ(output[0].size(), 1);
  EXPECT_EQ(output[1].size(), 2);
  EXPECT_EQ((output[1][1]->Is<format::CSC<TYPE>>()), true);
  delete output[1][1];
  EXPECT_EQ(output[2].size(), 2);
  EXPECT_EQ((output[2][1]->Is<format::CSR<TYPE>>()), true);
  delete output[2][1];
#undef ConversionPair
#undef TYPE
}

#define TYPE int, int, int
class ConversionChainFixture : public ::testing::Test {
 protected:
  struct FakeContext
      : utils::IdentifiableImplementation<FakeContext, context::Context> {
    virtual bool IsEquivalent(context::Context* i) const {
      return i->get_id() == this->get_id();
    };
  };
  void SetUp() override {
    csr = new sparsebase::format::CSR<int, int, int>(
        n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
    coo = new sparsebase::format::COO<int, int, int>(
        n, m, nnz, coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
    csc = new format::CSC<int, int, int>(n, m, csc_col_ptr, csc_row, csc_vals,
                                         sparsebase::format::kNotOwned);
    c.ClearConversionFunctions(csr->get_id(), coo->get_id(), false);
    c.ClearConversionFunctions(csr->get_id(), csc->get_id(), false);
    c.ClearConversionFunctions(coo->get_id(), csc->get_id(), false);
    c.ClearConversionFunctions(coo->get_id(), csr->get_id(), false);
    c.ClearConversionFunctions(csc->get_id(), csr->get_id(), false);
    c.ClearConversionFunctions(csc->get_id(), coo->get_id(), false);
    c.RegisterConversionFunction(
        csr->get_id(), coo->get_id(), returnCoo,
        [](context::Context*, context::Context* to) -> bool {
          return to->get_id() == context::CPUContext::get_id_static();
        });
    c.RegisterConversionFunction(
        coo->get_id(), csr->get_id(), returnCsr,
        [](context::Context* from, context::Context* to) -> bool {
          return to->get_id() == FakeContext::get_id_static();
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

TEST_F(ConversionChainFixture, SingleStep) {
  converter::ConversionSchema schema;
  std::vector<std::vector<format::Format*>> output;
  ConversionChain chain;
  format::Format* output_format;
  // single format
  // No conversion -- wrong context
  chain = c.GetConversionChain(coo->get_id(), &cpu_c, csr->get_id(), {&cpu_c},
                               false);
  EXPECT_EQ(chain.has_value(), false);
  // No conversion -- no function
  chain = c.GetConversionChain(csc->get_id(), &cpu_c, csr->get_id(), {&cpu_c},
                               false);
  EXPECT_EQ(chain.has_value(), false);
  // One conversion -- cpu context
  chain = c.GetConversionChain(csr->get_id(), &cpu_c, coo->get_id(), {&cpu_c},
                               false);
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
            context::CPUContext::get_id_static());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_id(), (format::COO<TYPE>::get_id_static()));
  delete output_format;
  // One conversion -- fake context
  chain = c.GetConversionChain(coo->get_id(), &cpu_c, csr->get_id(), {&fake_c},
                               false);
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
            FakeContext::get_id_static());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_id(), (format::CSR<TYPE>::get_id_static()));
  delete output_format;
  // Multiple conversions
}

TEST_F(ConversionChainFixture, MultiStep) {
  converter::ConversionSchema schema;
  std::vector<std::vector<format::Format*>> output;
  ConversionChain chain;
  format::Format* output_format;
  c.RegisterConversionFunction(
      coo->get_id(), csc->get_id(), returnCsc,
      [](context::Context* from, context::Context* to) -> bool {
        return to->get_id() == FakeContext::get_id_static();
      });
  // single format
  // No conversion -- only one needed context
  chain = c.GetConversionChain(csr->get_id(), &cpu_c, csc->get_id(), {&cpu_c},
                               false);
  EXPECT_EQ(chain.has_value(), false);
  // No conversion -- only one needed context
  chain = c.GetConversionChain(csr->get_id(), &cpu_c, csc->get_id(), {&fake_c},
                               false);
  EXPECT_EQ(chain.has_value(), false);
  // Two conversions -- only one needed context
  chain = c.GetConversionChain(csr->get_id(), &cpu_c, csc->get_id(),
                               {&fake_c, &cpu_c}, false);
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 2);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_id()),
            context::CPUContext::get_id_static());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_id(), (format::COO<TYPE>::get_id_static()));
  delete output_format;
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[1])->get_id()),
            FakeContext::get_id_static());
  output_format = (std::get<0>(std::get<0>(*chain)[1]))(nullptr, nullptr);
  EXPECT_EQ(output_format->get_id(), (format::CSC<TYPE>::get_id_static()));
  delete output_format;
}
TEST_F(ConversionChainFixture, CanConvert) {
  // single step
  // can
  EXPECT_TRUE(c.CanConvert(csr->get_id(), &cpu_c, coo->get_id(), &cpu_c));
  // can
  EXPECT_TRUE(
      c.CanConvert(csr->get_id(), &cpu_c, coo->get_id(), {&cpu_c, &fake_c}));
  // can't -- wrong context
  EXPECT_FALSE(c.CanConvert(csr->get_id(), &fake_c, coo->get_id(), &fake_c));
  // can -- same context and format
  EXPECT_TRUE(c.CanConvert(coo->get_id(), &fake_c, coo->get_id(), &fake_c));
  EXPECT_TRUE(
      c.CanConvert(coo->get_id(), &fake_c, coo->get_id(), {&cpu_c, &fake_c}));
  // can't -- different context but same format
  EXPECT_FALSE(c.CanConvert(coo->get_id(), &fake_c, coo->get_id(), &cpu_c));

  // multi-step
  c.RegisterConversionFunction(
      coo->get_id(), csc->get_id(), returnCsc,
      [](context::Context* from, context::Context* to) -> bool {
        return to->get_id() == FakeContext::get_id_static();
      });
  // can
  EXPECT_TRUE(
      c.CanConvert(csr->get_id(), &cpu_c, csc->get_id(), {&fake_c, &cpu_c}));
  // can't -- wrong contexts
  EXPECT_FALSE(
      c.CanConvert(csr->get_id(), &cpu_c, csc->get_id(), {&cpu_c, &cpu_c}));
  EXPECT_FALSE(
      c.CanConvert(csr->get_id(), &cpu_c, csc->get_id(), {&fake_c, &fake_c}));
}
#include "sparsebase/converter/converter_store.h"

TEST(ConverterStore, All) {
  using ConvStore = sparsebase::converter::ConverterStore;
  using ConvType =
      sparsebase::converter::ConverterOrderTwo<unsigned int, int, unsigned int>;
  // check the singleton-ness
  EXPECT_EQ(&ConvStore::GetStore(), &ConvStore::GetStore());
  // Check the reference counter value
  sparsebase::format::CSR<int, int, int> csr(
      n, m, csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  auto new_csr =
      csr.Convert<sparsebase::format::CSR, unsigned int, int, unsigned int>();
  auto ptr = ConvStore::GetStore().get_converter<ConvType>();
  EXPECT_EQ(ptr.use_count(), 2);
  auto ptr2 = ConvStore::GetStore().get_converter<ConvType>();
  EXPECT_EQ(ptr.use_count(), 3);
  EXPECT_EQ(ptr.get(), ptr2.get());
  std::weak_ptr<sparsebase::converter::Converter> w_ptr(ptr);
  ptr.reset();
  EXPECT_NE(w_ptr.lock(), nullptr);
  ptr2.reset();
  EXPECT_NE(w_ptr.lock(), nullptr);
  delete new_csr;
  EXPECT_EQ(w_ptr.lock(), nullptr);
}

#undef ConversionPair
#undef TYPE

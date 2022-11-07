#include <iostream>

#include "gtest/gtest.h"
#include "common.inc"
#include "sparsebase/sparsebase.h"
using namespace sparsebase;
using namespace converter;

#define CONVERT_AND_COMPARE_Array(input_array, VALT)                        \
  {                                                                         \
    auto output_array = (input_array)->Convert<format::Array, VALT>(false); \
    compare_arrays((input_array), output_array);                            \
    delete output_array;                                                    \
  }

#define MOVE_CONVERT_AND_COMPARE_Array(input_arr, VALT)                  \
  {                                                                      \
    auto copy = dynamic_cast<decltype(input_arr)>((input_arr)->Clone()); \
    auto output_arr = (copy)->Convert<format::Array, VALT>(true);        \
    compare_arrays((input_arr), output_arr);                             \
    check_move_arrays(copy, output_arr);                                 \
    delete output_arr;                                                   \
    delete copy;                                                         \
  }

template <typename ArrayType1, typename ArrayType2>
void compare_arrays(ArrayType1* array1, ArrayType2* array2) {
  auto nnz1 = array1->get_num_nnz();
  auto nnz2 = array2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(array1->get_dimensions()[0], array2->get_dimensions()[0]);
  compare_arrays(array1->get_vals(), array2->get_vals(), nnz1, "vals");
}

template <template <typename> typename ArrayType1,
    template <typename> typename ArrayType2, typename V1, typename V2>
void check_move_arrays(ArrayType1<V1>* array1, ArrayType2<V2>* array2) {
  auto nnz1 = array1->get_num_nnz();
  auto nnz2 = array2->get_num_nnz();
  ASSERT_EQ(nnz1, nnz2);
  ASSERT_EQ(array1->get_dimensions()[0], array2->get_dimensions()[0]);
  if constexpr (std::is_same_v<V1, V2>)
    EXPECT_EQ((void*)array1->get_vals(), (void*)array2->get_vals());
  else
    EXPECT_NE((void*)array1->get_vals(), (void*)array2->get_vals());
}


TEST(FormatOrderOneTypeConversion, Array) {
sparsebase::format::Array<int> array(nnz, coo_vals,
                                     sparsebase::format::kNotOwned);
CONVERT_AND_COMPARE_Array(&array, unsigned int);
float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
double coo_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
sparsebase::format::Array<float> array_f(nnz, coo_vals_f,
                                         sparsebase::format::kNotOwned);
CONVERT_AND_COMPARE_Array(&array_f, double);
// sparsebase::format::Array<double> array_d(
//    nnz, coo_vals_d, sparsebase::format::kNotOwned);
// CONVERT_AND_COMPARE_Array(&array_d,float);
}
TEST(FormatOrderOneTypeMoveConversion, Array) {
sparsebase::format::Array<int> array(nnz, coo_vals,
                                     sparsebase::format::kNotOwned);
MOVE_CONVERT_AND_COMPARE_Array(&array, unsigned int);
float coo_vals_f[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
double coo_vals_d[7] = {0, -1, 3, -2121234, 0.1231, -12312.12311, -6666};
sparsebase::format::Array<float> array_f(nnz, coo_vals_f,
                                         sparsebase::format::kNotOwned);
MOVE_CONVERT_AND_COMPARE_Array(&array_f, double);
// sparsebase::format::Array<double> array_d(
//    nnz, coo_vals_d, sparsebase::format::kNotOwned);
// CONVERT_AND_COMPARE_Array(&array_d,float);
}

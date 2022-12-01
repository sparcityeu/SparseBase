#include <iostream>

#include "common.inc"
#include "gtest/gtest.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/format_order_one.h"
TEST(FormatOrderOne, Convert) {
  sparsebase::format::Array<int> array(4, coo_vals,
                                       sparsebase::format::kNotOwned);

  sparsebase::context::CPUContext cpu_context;
  sparsebase::format::Array<int> *conv_arr =
      array.Convert<sparsebase::format::Array>(&cpu_context);
  EXPECT_EQ(conv_arr, &array);
  // Check the dimensions
  EXPECT_EQ(conv_arr->get_num_nnz(), 4);
  std::vector<sparsebase::format::DimensionType> expected_dimensions{4};
  EXPECT_EQ(conv_arr->get_dimensions(), expected_dimensions);

  // Check the array
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(conv_arr->get_vals()[i], coo_vals[i]);
  }
}

template <typename T>
class StubFormatOrderOne : sparsebase::utils::IdentifiableImplementation<StubFormatOrderOne<T>, sparsebase::format::FormatOrderOne<T>>{

 public:
  StubFormatOrderOne () {
    this->context_ = std::unique_ptr<sparsebase::context::Context>(new sparsebase::context::CPUContext);
  }
  sparsebase::format::Format *Clone() const { return nullptr; }
};
template <typename T>
class dummy{};

TEST(Is, FormatOrderOne){
  
  sparsebase::format::Array<int> array0(4, coo_vals,
                                       sparsebase::format::kNotOwned);
  ASSERT_TRUE(array0.Is<sparsebase::format::Array>());
  ASSERT_FALSE(array0.Is<StubFormatOrderOne>());
  ASSERT_FALSE(array0.Is<dummy>());
  EXPECT_FALSE(array0.Is<sparsebase::format::FormatOrderOne>());
}


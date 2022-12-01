#include <iostream>

#include "common.inc"
#include "gtest/gtest.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
using namespace sparsebase;
TEST(FormatImplementation, FormatID) {
  // Same Format type with different template parameters
  // should have a different id
  std::type_index id_csriii = format::CSR<int, int, int>::get_id_static();
  std::type_index id_csriif = format::CSR<int, int, float>::get_id_static();
  EXPECT_NE(id_csriii, id_csriif);

  // Different Format types should have different ids
  std::type_index id_cooiii = format::COO<int, int, int>::get_id_static();
  EXPECT_NE(id_csriii, id_cooiii);

  int csr_row_ptr[5]{0, 2, 3, 3, 4};
  int csr_col[4]{2, 0, 1, 3};
  sparsebase::format::CSR<int, int, int> csr(
      4, 4, csr_row_ptr, csr_col, nullptr, sparsebase::format::kNotOwned);

  // If the Format type and templates are the same
  // both the static and non-static function should give the same id
  std::type_index id_csriii_obj = csr.get_id();
  EXPECT_EQ(id_csriii, id_csriii_obj);
}

TEST(FormatImplementation, FormatName) {
  // Same Format type with different template parameters
  // should have a different id
  std::string name_csriii = format::CSR<int, int, int>::get_name_static();
  std::string name_csriif = format::CSR<int, int, float>::get_name_static();
  EXPECT_NE(name_csriii, name_csriif);

  // Different Format types should have different ids
  std::string name_cooiii = format::COO<int, int, int>::get_name_static();
  EXPECT_NE(name_csriii, name_cooiii);

  int csr_row_ptr[5]{0, 2, 3, 3, 4};
  int csr_col[4]{2, 0, 1, 3};
  sparsebase::format::CSR<int, int, int> csr(
      4, 4, csr_row_ptr, csr_col, nullptr, sparsebase::format::kNotOwned);

  // If the Format type and templates are the same
  // both the static and non-static function should give the same id
  std::string name_csriii_obj = csr.get_name();
  EXPECT_EQ(name_csriii, name_csriii_obj);

  std::string name_csriii_mangled = csr.get_id().name();
  EXPECT_NE(name_csriii, name_csriii_mangled);
}

format::COO<int, int, int> g_coo(4, 4, 4, coo_row, coo_col, coo_vals);
format::Format* dummy_conversion(format::Format* p, context::Context*) {
  return &g_coo;
}

TEST(FormatImplementation, SetConverter) {
  auto conv = std::make_shared<converter::ConverterOrderTwo<int, int, int>>();
  conv->ClearConversionFunctions();
  conv->RegisterConversionFunction(
      format::CSR<int, int, int>::get_id_static(),
      format::COO<int, int, int>::get_id_static(), dummy_conversion,
      [](context::Context*, context::Context*) { return true; });
  format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col, csr_vals);

  csr.set_converter(conv);

  EXPECT_EQ(csr.Convert<format::COO>((&csr)->get_context()), &g_coo);
}

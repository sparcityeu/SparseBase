#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/permute/permute_order_one.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::permute;
#include "../functionality_common.inc"
TEST(ArrayPermute, Basic) {
  context::CPUContext cpu_context;
  PermuteOrderOne<int, float> transform(inverse_perm_array);
  format::Format *inv_arr_fp =
      transform.GetPermutation(&orig_arr, {&cpu_context}, false);
  format::Array<float> *inv_arr =
      inv_arr_fp->AsAbsolute<format::Array<float>>();
  format::FormatOrderOne<float> *x = inv_arr;
  x->As<format::Array>();
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(inv_arr->get_vals()[i], reordered_array[i]);
  }
  EXPECT_NO_THROW(transform.GetPermutation(&orig_arr, {&cpu_context}, false));
}
TEST(ArrayPermute, Inverse) {
  context::CPUContext cpu_context;
  auto inv_p = ReorderBase::InversePermutation(inverse_perm_array,
                                               global_csr.get_dimensions()[0]);
  PermuteOrderOne<int, float> inverse_transform(inv_p);
  format::Format *inv_inversed_arr_fp =
      inverse_transform.GetPermutation(&inv_arr, {&cpu_context}, false);
  format::Array<float> *inv_inversed_arr =
      inv_inversed_arr_fp->AsAbsolute<format::Array<float>>();
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(inv_inversed_arr->get_vals()[i], original_array[i]);
  }
}

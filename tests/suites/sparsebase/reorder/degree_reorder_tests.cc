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
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
#include "../functionality_common.inc"

TEST(DegreeReorder, AscendingOrder) {
  DegreeReorder<int, int, int> reorder(true);
  auto order = reorder.GetReorder(&global_csr, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr);
}
TEST(DegreeReorder, DescendingOrder) {
  DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_csr, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
}
TEST(DegreeReorder, TwoParamsConversion) {
  DegreeReorder<int, int, int> reorder(false);
  EXPECT_THROW(reorder.GetReorder(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  try {
    reorder.GetReorder(&global_coo, {&cpu_context}, true);
  } catch (
      utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>
          &exception) {
    CompareVectorsOfTypeIndex(exception.used_format_,
                              {format::CSR<int, int, int>::get_id_static()});
    auto class_available_formats = reorder.GetAvailableFormats();
    auto returned_available_formats = exception.available_formats_;
    sort(class_available_formats.begin(), class_available_formats.end());
    sort(returned_available_formats.begin(), returned_available_formats.end());
    for (int i = 0; i < class_available_formats.size(); i++) {
      CompareVectorsOfTypeIndex(class_available_formats[i],
                                returned_available_formats[i]);
    }
  }
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
}
TEST(ReorderTypeTest, DescendingWithParams) {
  DegreeReorder<int, int, int> reorder(true);
  DegreeReorderParams param(false);
  auto order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
  EXPECT_NO_THROW(
      reorder.GetReorder(&global_csr, &param, {&cpu_context}, true));
  order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
}
TEST(ReorderTypeTest, AscendingWithParams) {
  DegreeReorder<int, int, int> reorder(false);
  DegreeReorderParams param(true);
  auto order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
  EXPECT_NO_THROW(
      reorder.GetReorder(&global_csr, &param, {&cpu_context}, true));
  order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
}

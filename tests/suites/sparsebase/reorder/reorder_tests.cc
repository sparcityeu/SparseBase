#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
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

TEST(ReorderTypeTest, NoCachConversion) {
  DegreeReorder<int, int, int> reorder(false);
  DegreeReorderParams param(true);
  auto order = reorder.GetReorder(&global_coo, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
  EXPECT_NO_THROW(
      reorder.GetReorder(&global_coo, &param, {&cpu_context}, true));
  order = reorder.GetReorder(&global_coo, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
}

TEST(ReorderTypeTest, CachedNoConversion) {
  DegreeReorder<int, int, int> reorder(false);
  DegreeReorderParams param(true);
  auto order =
      reorder.GetReorderCached(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, true);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_EQ(std::get<0>(order)[0].size(), 0);
  EXPECT_NO_THROW(
      reorder.GetReorderCached(&global_csr, &param, {&cpu_context}, true));
  order = reorder.GetReorderCached(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, true);
}

TEST(ReorderTypeTest, CachedConversionTwoParams) {
  DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorderCached(&global_coo, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, false);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_EQ(std::get<0>(order)[0].size(), 1);
  EXPECT_NE(std::get<0>(order)[0][0], nullptr);
  auto cached_csr =
      std::get<0>(order)[0][0]->AsAbsolute<format::CSR<int, int, int>>();
  compare_csr(&global_csr, cached_csr);
  EXPECT_THROW(reorder.GetReorderCached(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  try {
    reorder.GetReorderCached(&global_coo, {&cpu_context}, true);
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
}

TEST(ReorderTypeTest, CachedNoConversionTwoParams) {
  DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorderCached(&global_csr, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, false);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_EQ(std::get<0>(order)[0].size(), 0);
  EXPECT_NO_THROW(reorder.GetReorderCached(&global_csr, {&cpu_context}, true));
  order = reorder.GetReorderCached(&global_csr, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, false);
}

TEST(ReorderTypeTest, CachedConversion) {
  DegreeReorder<int, int, int> reorder(false);
  DegreeReorderParams param(true);
  auto order =
      reorder.GetReorderCached(&global_coo, &param, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, true);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_NE(std::get<0>(order)[0][0], nullptr);
  auto cached_csr =
      std::get<0>(order)[0][0]->AsAbsolute<format::CSR<int, int, int>>();
  compare_csr(&global_csr, cached_csr);
  EXPECT_THROW(
      reorder.GetReorderCached(&global_coo, &param, {&cpu_context}, false),
      utils::DirectExecutionNotAvailableException<
          std::vector<std::type_index>>);
  try {
    reorder.GetReorderCached(&global_coo, &param, {&cpu_context}, true);
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
}

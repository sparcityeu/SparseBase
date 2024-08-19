#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/triangle_count.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"

using namespace sparsebase;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::feature;
#include "../functionality_common.inc"

class TriangleCountTest : public ::testing::Test {
};

TEST_F(TriangleCountTest, DirectedTriangleTests) {
  const int n_ = 10, m_ = 10;
  int row_ptr_[n_+1] = {0, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12};
  int col_[12] = {2, 3, 1, 6, 4, 5, 8, 9, 7, 9, 7, 8};
  /* Triangles
   * 1 -> 2 -> 3 -> 1,
   * 4 -> 6 -> 5 -> 4,
   * 7 -> 8 -> 9 -> 7,
   * 7 -> 9 -> 8 -> 7
  */
  int64_t ans = 4;
  auto csr = new sparsebase::format::CSR<int, int, void>(n_, m_, row_ptr_, col_, nullptr,
                                                         sparsebase::format::kOwned);

  TriangleCountParams p_directed(true);
  auto feature =  feature::TriangleCount<int,int,void>(p_directed);
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 1);
  EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

  // Test get_subs
  auto subs = feature.get_subs();
  // a single sub-feature
  EXPECT_EQ(subs.size(), 1);
  // same type as feature but different address
  auto &feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), std::type_index(typeid(feature)));
  EXPECT_NE(subs[0], &feature);

  // Check GetTriangleCountCSR implementation function
  auto triangle_count =
      feature::TriangleCount<int, int, void>::GetTriangleCountCSR({csr}, &p_directed);

  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;
  // Check GetTriangleCount
  triangle_count = feature.GetTriangleCount(csr, {&cpu_context}, true);
  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;

  triangle_count = feature.GetTriangleCount(csr, {&cpu_context}, false);
  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;

  // Check GetTriangleCount with conversion
  triangle_count = feature.GetTriangleCount(csr, {&cpu_context}, true);
  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;

  // Check Extract
  auto feature_map = feature.Extract(csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int64_t *>(feature_map[feature.get_id()]), ans);
}

TEST_F(TriangleCountTest, UndirectedTriangleTests) {
  const int n_ = 10, m_ = 10;
  int row_ptr_[n_+1] = {0, 0, 2, 4, 6, 8, 10, 12, 13, 15, 16};
  int col_[16] = {2, 3, 1, 3, 1, 2, 5, 6, 4, 6, 4, 5, 8, 7, 9, 8};
  /* Triangles
   * (1, 2, 3)
   * (4, 6, 5)
   */
  int64_t ans = 2;
  auto csr = new sparsebase::format::CSR<int, int, void>(n_, m_, row_ptr_, col_, nullptr,
                                                         sparsebase::format::kOwned);

  TriangleCountParams p_undirected(false);
  auto feature =  feature::TriangleCount<int,int,void>(p_undirected);
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 1);
  EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

  // Test get_subs
  auto subs = feature.get_subs();
  // a single sub-feature
  EXPECT_EQ(subs.size(), 1);
  // same type as feature but different address
  auto &feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), std::type_index(typeid(feature)));
  EXPECT_NE(subs[0], &feature);

  // Check GetTriangleCountCSR implementation function
  auto triangle_count =
      feature::TriangleCount<int, int, void>::GetTriangleCountCSR({csr}, &p_undirected);

  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;
  // Check GetTriangleCount
  triangle_count = feature.GetTriangleCount(csr, {&cpu_context}, true);
  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;

  triangle_count = feature.GetTriangleCount(csr, {&cpu_context}, false);
  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;

  // Check GetTriangleCount with conversion
  triangle_count = feature.GetTriangleCount(csr, {&cpu_context}, true);
  EXPECT_EQ(*triangle_count, ans);
  delete triangle_count;

  // Check Extract
  auto feature_map = feature.Extract(csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int64_t *>(feature_map[feature.get_id()]), ans);
}
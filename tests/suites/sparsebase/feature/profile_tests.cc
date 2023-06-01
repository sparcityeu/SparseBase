#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/profile.h"
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

class ProfileTest : public ::testing::Test {
 protected:
  feature::Profile<int, int, void> feature;
  
  struct Params1 : sparsebase::utils::Parameters {};
};

TEST_F(ProfileTest, AllTests) {
  sparsebase::context::CPUContext cpu_context;
  Params1 p1;
  const int n = 7, m = 7;
  const int nnz = 12;
  int row_ptr_[n+1] = {0, 2, 2, 5, 7, 9, 11, 12};
  int col_[nnz] = {2, 3, 0, 3, 4, 0, 2, 2, 5, 4, 6, 5};
  /*     0 1 2 3 4 5 6
   * 0   0 0 1 1 0 0 0
   * 1   0 0 0 0 0 0 0
   * 2   1 0 0 1 1 0 0
   * 3   1 0 1 0 0 0 0
   * 4   0 0 1 0 0 1 0
   * 5   0 0 0 0 1 0 1
   * 6   0 0 0 0 0 1 0 */

  int ans = 9;
  auto csr = new sparsebase::format::CSR<int, int, void>(n, m, row_ptr_, col_, nullptr,
                                                         sparsebase::format::kOwned);
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

  // Check GetProfileCSR implementation function
  int* profile =
      feature::Profile<int, int, void>::GetProfileCSR({csr}, &p1);

  EXPECT_EQ(*profile, ans);
  delete profile;

  // Check GetProfile
  profile = feature.GetProfile(csr, {&cpu_context}, true);
  EXPECT_EQ(*profile, ans);
  delete profile;

  profile = feature.GetProfile(csr, {&cpu_context}, false);
  EXPECT_EQ(*profile, ans);
  delete profile;

  // Check Extract
  auto feature_map = feature.Extract(csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int *>(feature_map[feature.get_id()]), ans);
}

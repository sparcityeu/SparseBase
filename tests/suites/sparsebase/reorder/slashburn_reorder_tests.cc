#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/reorder/gray_reorder.h"
#include "sparsebase/reorder/slashburn_reorder.h"
#include "sparsebase/reorder/reorderer.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
TEST(SlashburnReorderTest, BasicTest) {
  SlashburnReorder<int, int, int> reorder(1, false, false);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  check_reorder(order, n);
}
TEST(SlashburnReorderTest, BasicTestMultiK) {
  SlashburnReorder<int, int, int> reorder(10, false, false);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  check_reorder(order, n);
}
TEST(SlashburnReorderTest, BasicTestGreedyHub) {
  SlashburnReorder<int, int, int> reorder(1, true, true);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  check_reorder(order, n);
}

#include <iostream>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>
#include <memory>

#include "gtest/gtest.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/coo.h"

#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/reorder/amd_reorder.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/utils/exception.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
TEST(AMDReorderTest, BasicTestBitSize16) {
  AMDReorderParams p;
AMDReorder<int, int, int> reorder(p);
auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
check_reorder(order, n);
}

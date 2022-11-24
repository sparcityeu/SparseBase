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

#include "sparsebase/utils/utils.h"
using namespace sparsebase;
TEST(TypeIndexHash, Basic) {
utils::TypeIndexVectorHash hasher;
// Empty vector
std::vector<std::type_index> vec;
EXPECT_EQ(hasher(vec), 0);
// Vector with values
vec.push_back(typeid(int));
vec.push_back(typeid(double));
vec.push_back(typeid(float));
size_t hash = 0;
for (auto tid : vec) {
hash += tid.hash_code();
}
EXPECT_EQ(hash, hasher(vec));
}

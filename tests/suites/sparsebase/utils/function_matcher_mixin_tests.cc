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

#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/function_matcher_mixin.h"
#include "sparsebase/utils/exception.h"
using namespace sparsebase;
using namespace sparsebase::utils;
using namespace sparsebase::context;
using namespace sparsebase::preprocess;
class FunctionMatcherMixinTest : public ::testing::Test {
 protected:
  GenericPreprocessType<int> concrete_preprocess;

  static int OneImplementationFunction(std::vector<format::Format *> inputs,
                                       utils::Parameters *) {
    return 1;
  }
  static int TwoImplementationFunction(std::vector<format::Format *> inputs,
                                       utils::Parameters *) {
    return 2;
  }
  static int ThreeImplementationFunction(std::vector<format::Format *> inputs,
                                         utils::Parameters *) {
    return 3;
  }
  static int FourImplementationFunction(std::vector<format::Format *> inputs,
                                        utils::Parameters *) {
    return 4;
  }
  struct GenericParams : utils::Parameters {
    GenericParams(int p) : param(p) {}
    int param;
  };
  FunctionMatcherMixinTest(){
    row_ptr = new int[n+1] {0, 2, 3, 4};
    cols = new int[nnz] {1, 2, 0, 0};
    rows = new int[nnz] {0, 0, 1, 2};
    vals = new int[nnz] {1, 2, 3, 4};
    csr = new format::CSR<int, int, int>(n, n, row_ptr, cols, vals,
                                         format::kNotOwned);
  }
  virtual ~FunctionMatcherMixinTest(){
    delete [] row_ptr;
    delete [] cols;
    delete [] rows;
    delete [] vals;
    delete csr;
  }
  const int n = 3;
  const int nnz = 4;
  int *row_ptr;
  int *cols;
  int *rows;
  int *vals;
  format::CSR<int, int, int>* csr;
};

TEST_F(FunctionMatcherMixinTest, BlackBox) {
  CPUContext cpu_context;
// Check calling with an empty map
EXPECT_THROW(
    concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
utils::FunctionNotFoundException);
EXPECT_THROW(
    concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
utils::FunctionNotFoundException);

// Check calling with no conversion needed
concrete_preprocess.RegisterFunction({csr->get_id()},
OneImplementationFunction);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
1);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
1);

// Check unregistering
EXPECT_EQ(
    concrete_preprocess.UnregisterFunction(
{sparsebase::format::CSR<int, int, int>::get_id_static()}),
true);
EXPECT_THROW(
    concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
utils::FunctionNotFoundException);
EXPECT_THROW(
    concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
utils::FunctionNotFoundException);

// Check unregistering an already unregistered key
EXPECT_EQ(
    concrete_preprocess.UnregisterFunction(
{sparsebase::format::CSR<int, int, int>::get_id_static()}),
false);


// Check calling with no conversion needed even though one is possible
concrete_preprocess.RegisterFunction({csr->get_id()},
OneImplementationFunction);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
1);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
1);

// Checking override
// Override an existing function in the map
concrete_preprocess.RegisterFunction({csr->get_id()},
ThreeImplementationFunction);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
3);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
3);

// Try to override but fail
EXPECT_EQ(concrete_preprocess.RegisterFunctionNoOverride(
{csr->get_id()}, FourImplementationFunction),
false);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
3);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
3);

// Try to override and succeed
concrete_preprocess.UnregisterFunction(
{sparsebase::format::CSR<int, int, int>::get_id_static()});
EXPECT_EQ(concrete_preprocess.RegisterFunctionNoOverride(
{csr->get_id()}, FourImplementationFunction),
true);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
4);
EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
4);

// Checking cached getters
// No conversion needed to be done
auto tup =
    concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, true);
EXPECT_EQ(std::get<0>(tup)[0].size(), 0);
EXPECT_EQ(std::get<1>(tup), 4);
tup =
concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, false);
EXPECT_EQ(std::get<0>(tup)[0].size(), 0);
EXPECT_EQ(std::get<1>(tup), 4);

// One conversion is done
concrete_preprocess.UnregisterFunction(
{sparsebase::format::CSR<int, int, int>::get_id_static()});
concrete_preprocess.RegisterFunction(
{sparsebase::format::COO<int, int, int>::get_id_static()}, TwoImplementationFunction);
auto tup2 =
    concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, true);
ASSERT_NE(std::get<0>(tup2)[0][0], nullptr);
ASSERT_NE(std::get<0>(tup2)[0][0]->get_id(), csr->get_id());
EXPECT_EQ(std::get<1>(tup2), 2);
EXPECT_THROW(
    concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
}

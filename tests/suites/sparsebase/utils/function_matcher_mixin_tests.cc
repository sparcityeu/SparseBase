#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/function_matcher_mixin.h"
using namespace sparsebase;
using namespace sparsebase::utils;
using namespace sparsebase::context;
;
#include "../functionality_common.inc"

template <typename ReturnType>
class GenericPreprocessType : public utils::FunctionMatcherMixin<ReturnType> {
 protected:
 public:
  int GetOutput(format::Format *csr, utils::Parameters *params,
                std::vector<context::Context *>, bool convert_input);
  std::tuple<std::vector<std::vector<format::Format *>>, int> GetOutputCached(
      format::Format *csr, utils::Parameters *params,
      std::vector<context::Context *> contexts, bool convert_input);
  virtual ~GenericPreprocessType();
};
template <typename ReturnType>
GenericPreprocessType<ReturnType>::~GenericPreprocessType() = default;

template <typename ReturnType>
int GenericPreprocessType<ReturnType>::GetOutput(
    format::Format *format, utils::Parameters *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->Execute(params, contexts, convert_input, format);
}

template <typename ReturnType>
std::tuple<std::vector<std::vector<format::Format *>>, int>
GenericPreprocessType<ReturnType>::GetOutputCached(
    format::Format *format, utils::Parameters *params,
    std::vector<context::Context *> contexts, bool convert_input) {
  return this->CachedExecute(params, contexts, convert_input, false, format);
}
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
  FunctionMatcherMixinTest() {
    row_ptr = new int[n + 1]{0, 2, 3, 4};
    cols = new int[nnz]{1, 2, 0, 0};
    rows = new int[nnz]{0, 0, 1, 2};
    vals = new int[nnz]{1, 2, 3, 4};
    csr = new format::CSR<int, int, int>(n, n, row_ptr, cols, vals,
                                         format::kNotOwned);
  }
  virtual ~FunctionMatcherMixinTest() {
    delete[] row_ptr;
    delete[] cols;
    delete[] rows;
    delete[] vals;
    delete csr;
  }
  const int n = 3;
  const int nnz = 4;
  int *row_ptr;
  int *cols;
  int *rows;
  int *vals;
  format::CSR<int, int, int> *csr;
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
  EXPECT_EQ(concrete_preprocess.UnregisterFunction(
                {sparsebase::format::CSR<int, int, int>::get_id_static()}),
            true);
  EXPECT_THROW(
      concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
      utils::FunctionNotFoundException);
  EXPECT_THROW(
      concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
      utils::FunctionNotFoundException);

  // Check unregistering an already unregistered key
  EXPECT_EQ(concrete_preprocess.UnregisterFunction(
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
      {sparsebase::format::COO<int, int, int>::get_id_static()},
      TwoImplementationFunction);
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

class MultiFormatKeyPreprocess : public utils::FunctionMatcherMixin<int> {
 public:
  std::tuple<std::vector<std::vector<format::Format *>>, int> GetCached(
      format::Format *f1, format::Format *f2, format::Format *f3,
      std::vector<context::Context *> contexts, bool convert_input,
      bool clear_intermediate) {
    auto p = new utils::Parameters;
    auto res = this->CachedExecute(p, std::move(contexts), convert_input,
                                   clear_intermediate, f1, f2, f3);
    return res;
  }
  MultiFormatKeyPreprocess() {
    this->RegisterFunction({format::CSR<int, int, int>::get_id_static(),
                            format::CSR<int, int, int>::get_id_static(),
                            format::CSR<int, int, int>::get_id_static()},
                           CSR_CSR_CSR);
    this->RegisterFunction({format::CSR<int, int, int>::get_id_static(),
                            format::CSC<int, int, int>::get_id_static(),
                            format::CSC<int, int, int>::get_id_static()},
                           CSR_CSC_CSC);
  }

 private:
  static int CSR_CSR_CSR(std::vector<format::Format *>, utils::Parameters *) {
    return 1;
  }
  static int CSR_CSC_CSC(std::vector<format::Format *>, utils::Parameters *) {
    return 1;
  }
};

TEST(MultiKeyFunctionMatcherMixinTest, MultiFormatKey) {
#define TYPE int, int, int
  MultiFormatKeyPreprocess x;
  auto c = std::make_shared<converter::ConverterOrderTwo<int, int, int>>();
  c->ClearConversionFunctions(format::CSR<int, int, int>::get_id_static(),
                              format::CSC<int, int, int>::get_id_static(),
                              false);
  context::CPUContext cpu;
  format::CSR<TYPE> *csr = global_csr.Clone()->AsAbsolute<format::CSR<TYPE>>();
  format::COO<TYPE> *coo = global_coo.Clone()->AsAbsolute<format::COO<TYPE>>();
  auto *csc = global_coo.Convert<format::CSC>();
  csr->set_converter(c);
  coo->set_converter(c);
  csc->set_converter(c);
  // No conversions needed on all three
  auto output = x.GetCached(csr, csr, csr, {&cpu}, true, false);
  auto intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for first only
  output = x.GetCached(coo, csr, csr, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 1);
  EXPECT_EQ((intermediate[0][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second only
  output = x.GetCached(csr, coo, csr, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second and third
  output = x.GetCached(csr, coo, coo, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 1);
  EXPECT_EQ((intermediate[2][0]->Is<format::CSC<TYPE>>()), true);
  // Conversion for second two-step
  output = x.GetCached(csr, csr, csc, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  ASSERT_EQ(intermediate[1].size(), 2);
  EXPECT_EQ((intermediate[1][0]->Is<format::COO<TYPE>>()), true);
  EXPECT_EQ((intermediate[1][1]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);
  delete csc;
#undef TYPE
}

TEST(MultiKeyFunctionMatcherMixinTest, MultiFormatKeyClearIntermediate) {
#define TYPE int, int, int
  MultiFormatKeyPreprocess x;
  auto c = std::make_shared<converter::ConverterOrderTwo<int, int, int>>();
  c->ClearConversionFunctions(format::CSR<int, int, int>::get_id_static(),
                              format::CSC<int, int, int>::get_id_static(),
                              false);
  context::CPUContext cpu;
  format::CSR<TYPE> *csr = global_csr.Clone()->AsAbsolute<format::CSR<TYPE>>();
  format::COO<TYPE> *coo = global_coo.Clone()->AsAbsolute<format::COO<TYPE>>();
  auto *csc = global_coo.Convert<format::CSC>();
  csr->set_converter(c);
  coo->set_converter(c);
  csc->set_converter(c);
  // No conversions needed on all three
  auto output = x.GetCached(csr, csr, csr, {&cpu}, true, true);
  auto intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for first only
  output = x.GetCached(coo, csr, csr, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 1);
  EXPECT_EQ((intermediate[0][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second only
  output = x.GetCached(csr, coo, csr, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second and third
  output = x.GetCached(csr, coo, coo, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 1);
  EXPECT_EQ((intermediate[2][0]->Is<format::CSC<TYPE>>()), true);
  // Conversion for second two-step
  output = x.GetCached(csr, csr, csc, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  ASSERT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);

  delete csc;
#undef TYPE
}

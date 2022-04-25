#include "gtest/gtest.h"
#include "sparsebase/config.h"

#include <iostream>
#include "sparsebase/format/format.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/context/context.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/converter/converter.h"
#include <typeindex>
#include <typeinfo>
#include <vector>
#include <tuple>
using namespace sparsebase;
using namespace sparsebase::preprocess;
TEST(TypeIndexHash, Basic){
  TypeIndexVectorHash hasher;
  // Empty vector
  std::vector<std::type_index> vec;
  EXPECT_EQ(hasher(vec), 0);
  // Vector with values
  vec.push_back(typeid(int));
  vec.push_back(typeid(double));
  vec.push_back(typeid(float));
  size_t hash = 0; 
  for (auto tid : vec){
    hash+= tid.hash_code();
  }
  EXPECT_EQ(hash, hasher(vec));
}

TEST(ConverterMixin, Basics){
  ConverterMixin<PreprocessType> instance;
  //! Check getting an empty converter
  ASSERT_EQ(instance.GetConverter(), nullptr);

  //! Check setting a converter
  utils::converter::ConverterOrderOne<int> converter;
  instance.SetConverter(converter);
  // Same type
  ASSERT_NE(dynamic_cast<utils::converter::ConverterOrderOne<int>*>(instance.GetConverter().get()), nullptr);
  // Different object
  ASSERT_NE(instance.GetConverter().get(), &converter);

  //! Check resetting a converter
  instance.ResetConverter();
  // Same type
  ASSERT_NE(dynamic_cast<utils::converter::ConverterOrderOne<int>*>(instance.GetConverter().get()), nullptr);
  // Different object
  ASSERT_NE(instance.GetConverter().get(), &converter);
}

class FunctionMatcherMixinTest : public ::testing::Test {
  protected:
    GenericPreprocessType<int> concrete_preprocess;
    sparsebase::format::CSR<int, int, int>* csr;
    int xadj[4] = {0, 2, 3, 4};
    int adj[4] = {1,2,0,0};
    sparsebase::context::CPUContext cpu_context;
  
  void SetUp() override{
    csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr);
  }

  static int OneImplementationFunction(std::vector<format::Format*> inputs, PreprocessParams*){
    return 1;
  }
  static int TwoImplementationFunction(std::vector<format::Format*> inputs, PreprocessParams*){
    return 2;
  }
  static int ThreeImplementationFunction(std::vector<format::Format*> inputs, PreprocessParams*){
    return 3;
  }
  static int FourImplementationFunction(std::vector<format::Format*> inputs, PreprocessParams*){
    return 4;
  }
  struct GenericParams : PreprocessParams {
    GenericParams(int p): param(p){}
    int param;
  };
};

TEST_F(FunctionMatcherMixinTest, BlackBox){
  // Check calling with an empty map
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), utils::FunctionNotFoundException);

  // Check calling with no conversion needed
  concrete_preprocess.RegisterFunction({csr->get_format_id()}, OneImplementationFunction);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), 1);

  // Check unregistering
  EXPECT_EQ(concrete_preprocess.UnregisterFunction({sparsebase::format::CSR<int, int, int>::get_format_id_static()}), true);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), utils::FunctionNotFoundException);

  // Check unregistering an already unregistered key
  EXPECT_EQ(concrete_preprocess.UnregisterFunction({sparsebase::format::CSR<int, int, int>::get_format_id_static()}), false);

  // Check calling with one conversion needed and no converter registered
  concrete_preprocess.RegisterFunction({sparsebase::format::COO<int, int, int>::get_format_id_static()}, TwoImplementationFunction);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), utils::NoConverterException);

  // Check calling with one conversion needed and a converter registered
  concrete_preprocess.SetConverter(utils::converter::ConverterOrderTwo<int, int, int>{});
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), 2);

  // Check calling with no conversion needed even though one is possible
  concrete_preprocess.RegisterFunction({csr->get_format_id()}, OneImplementationFunction);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), 1);

  // Checking override
  // Override an existing function in the map
  concrete_preprocess.RegisterFunction({csr->get_format_id()}, ThreeImplementationFunction);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), 3);

  // Try to override but fail
  EXPECT_EQ(concrete_preprocess.RegisterFunctionNoOverride({csr->get_format_id()}, FourImplementationFunction), false);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), 3);

  // Try to override and succeed
  concrete_preprocess.UnregisterFunction({sparsebase::format::CSR<int, int, int>::get_format_id_static()});
  EXPECT_EQ(concrete_preprocess.RegisterFunctionNoOverride({csr->get_format_id()}, FourImplementationFunction), true);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}), 4);

  // Checking cached getters
  // No conversion needed to be done
  auto tup = concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context});
  EXPECT_EQ(std::get<0>(tup)[0], nullptr);
  EXPECT_EQ(std::get<1>(tup), 4);

  // One conversion is done
  concrete_preprocess.UnregisterFunction({sparsebase::format::CSR<int, int, int>::get_format_id_static()});
  auto tup2 = concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context});
  ASSERT_NE(std::get<0>(tup2)[0], nullptr);
  ASSERT_NE(std::get<0>(tup2)[0]->get_format_id(), csr->get_format_id());
  EXPECT_EQ(std::get<1>(tup2), 2);
}

TEST(DegreeReorder, AscendingOrder){
  int xadj[4] = {0, 2, 3, 4};
  int adj[4] = {1,2,0,0};
  sparsebase::context::CPUContext cpu_context;
  sparsebase::format::CSR<int, int, int> csr(3, 3, xadj, adj, nullptr);
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(true);
  auto order = reorder.GetReorder(&csr, {&cpu_context});
  for (int i =0; i< 2; i++){
    auto u = order[i];
    auto v = order[i+1];
    EXPECT_GE(xadj[v+1]-xadj[v], xadj[u+1]-xadj[u]);
  }
}
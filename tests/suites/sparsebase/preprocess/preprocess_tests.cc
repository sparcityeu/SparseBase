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
class Degrees_DegreeDistributionTest : public ::testing::Test {
protected:
    Degrees_DegreeDistribution<int, int, int, float> feature;
    sparsebase::format::CSR<int, int, int>* csr;
    int xadj[4] = {0, 2, 3, 4};
    int adj[4] = {1,2,0,0};
    sparsebase::context::CPUContext cpu_context;

    void SetUp() override{
        csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr);
    }
    struct Params1 : sparsebase::preprocess::PreprocessParams{};
    struct Params2 : sparsebase::preprocess::PreprocessParams{};
};

TEST_F(Degrees_DegreeDistributionTest, FeaturePreprocessTypeTests){
    std::shared_ptr<Params1> p1(new Params1);
    std::shared_ptr<Params2> p2(new Params2);
    // Check getting feature id
    EXPECT_EQ(std::type_index(typeid(Degrees_DegreeDistribution<int, int, int, float>)), feature.get_feature_id());
    // Getting a params object for an unset params
    EXPECT_THROW(feature.get_params(Degrees<int, int, int>::get_feature_id_static()).get(), utils::FeatureParamsException);
    // Checking setting params of a sub-feature
    feature.set_params(Degrees<int, int, int>::get_feature_id_static(), p1);
    EXPECT_EQ(feature.get_params(Degrees<int, int, int>::get_feature_id_static()).get(), p1.get());
    EXPECT_NE(feature.get_params(Degrees<int, int, int>::get_feature_id_static()).get(), p2.get());
    // Checking setting params of feature that isn't a sub-feature
    EXPECT_THROW(feature.set_params(typeid(int), p1), utils::FeatureParamsException);
}
class DegreesTest : public ::testing::Test {
protected:
    Degrees<int, int, int> feature;
    sparsebase::format::CSR<int, int, int>* csr;
    int xadj[4] = {0, 2, 3, 4};
    int adj[4] = {1,2,0,0};
    int degrees[3] = {2, 1, 1};
    sparsebase::context::CPUContext cpu_context;

    void SetUp() override{
        csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr);
    }
    struct Params1 : sparsebase::preprocess::PreprocessParams{};
    struct Params2 : sparsebase::preprocess::PreprocessParams{};
};

TEST_F(DegreesTest, AllTests){
    // test get_sub_ids
    EXPECT_EQ(feature.get_sub_ids().size(), 1);
    EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

    // Test get_subs
    auto subs = feature.get_subs();
    // a single sub-feature
    EXPECT_EQ(subs.size(), 1);
    // same type as feature but different address
    EXPECT_EQ(std::type_index(typeid(*(subs[0]))), std::type_index(typeid(feature)));
    EXPECT_NE(subs[0], &feature);

    // Check GetDegreesCSR implementation function
    Params1 p1;
    auto degrees_array = Degrees<int, int, int>::GetDegreesCSR({csr}, &p1);
    for (int i =0; i<3; i++){
        EXPECT_EQ(degrees_array[i], degrees[i]);
    }
    delete [] degrees_array;
    // Check GetDegrees
    degrees_array = feature.GetDegrees({csr}, {&cpu_context});
    for (int i =0; i<3; i++){
        EXPECT_EQ(degrees_array[i], degrees[i]);
    }
    // Check Extract
    auto feature_map = feature.Extract(csr, {&cpu_context});
    // Check map size and type
    EXPECT_EQ(feature_map.size(), 1);
    for (auto feat : feature_map){
        EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
    }
    for (int i =0; i<3; i++){
        EXPECT_EQ(std::any_cast<int*>(feature_map[feature.get_feature_id()])[i], degrees[i]);
    }
}
class DegreeDistributionTest : public ::testing::Test {
protected:
    DegreeDistribution<int, int, int, float> feature;
    sparsebase::format::CSR<int, int, int>* csr;
    sparsebase::format::COO<int, int, int>* coo;
    int xadj[4] = {0, 2, 3, 4};
    int is[4] = {0, 0, 1, 2};
    int adj[4] = {1,2,0,0};
    float distribution[3] = {2.0/4, 1.0/4, 1.0/4};
    sparsebase::context::CPUContext cpu_context;

    void SetUp() override{
        csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr);
        coo = new sparsebase::format::COO<int, int, int>(3, 3, 4, is, adj, nullptr);
    }
    struct Params1 : sparsebase::preprocess::PreprocessParams{};
    struct Params2 : sparsebase::preprocess::PreprocessParams{};
};

TEST_F(DegreeDistributionTest, AllTests){
    // test get_sub_ids
    EXPECT_EQ(feature.get_sub_ids().size(), 1);
    EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

    // Test get_subs
    auto subs = feature.get_subs();
    // a single sub-feature
    EXPECT_EQ(subs.size(), 1);
    // same type as feature but different address
    EXPECT_EQ(std::type_index(typeid(*(subs[0]))), std::type_index(typeid(feature)));
    EXPECT_NE(subs[0], &feature);

    // Check GetDegreeDistributionCSR implementation function
    Params1 p1;
    auto distribution_array = DegreeDistribution<int, int, int, float>::GetDegreeDistributionCSR({csr}, &p1);
    for (int i =0; i<3; i++){
    EXPECT_EQ(distribution_array[i], distribution[i]);
    }
    delete [] distribution_array ;
    //// Check GetDistribution (function matcher)
    distribution_array = feature.GetDistribution(csr, {&cpu_context});
    for (int i =0; i<3; i++){
    EXPECT_EQ(distribution_array[i], distribution[i]);
    }
    delete [] distribution_array ;
    // Check GetDistribution with conversion
    distribution_array = feature.GetDistribution(coo, {&cpu_context});
    for (int i =0; i<3; i++){
        EXPECT_EQ(distribution_array[i], distribution[i]);
    }
    delete [] distribution_array ;
    // Check GetDistribution with conversion and cached
    auto distribution_array_format = feature.GetDistributionCached(coo, {&cpu_context});
    for (int i =0; i<3; i++){
        EXPECT_EQ(get<1>(distribution_array_format)[i], distribution[i]);
    }
    delete [] get<1>(distribution_array_format);
    auto cached_data = get<0>(distribution_array_format);
    ASSERT_EQ(cached_data.size(), 1);
    ASSERT_EQ(cached_data[0]->get_format_id(), std::type_index(typeid(*csr)));
    auto converted_csr = cached_data[0]->As<format::CSR<int, int, int>>();
    auto row_ptr = converted_csr->get_row_ptr();
    auto col = converted_csr->get_col();
    for (int i =0; i<4;i++){
        EXPECT_EQ(row_ptr[i], csr->get_row_ptr()[i]);
    }
    for (int i =0; i<4;i++) {
        EXPECT_EQ(col[i], csr->get_col()[i]);
    }
    // Check Extract
    auto feature_map = feature.Extract(csr, {&cpu_context});
    // Check map size and type
    EXPECT_EQ(feature_map.size(), 1);
    for (auto feat : feature_map){
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
    }
    for (int i =0; i<3; i++){
    EXPECT_EQ(std::any_cast<float*>(feature_map[feature.get_feature_id()])[i], distribution[i]);
    }
}

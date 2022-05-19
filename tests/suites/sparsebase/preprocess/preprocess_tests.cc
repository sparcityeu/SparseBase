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
int xadj[4] = {0, 2, 3, 4};
int adj[4] = {1,2,0,0};
int is[4] = {0, 0, 1, 2};
float distribution[3] = {2.0/4, 1.0/4, 1.0/4};
int degrees[3] = {2, 1, 1};
sparsebase::context::CPUContext cpu_context;
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

  void SetUp() override{
    csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr, sparsebase::format::kNotOwned);
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
  void TearDown() override{
      delete csr;
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
  sparsebase::context::CPUContext cpu_context;
  sparsebase::format::CSR<int, int, int> csr(3, 3, xadj, adj, nullptr, sparsebase::format::kNotOwned);
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(true);
  auto order = reorder.GetReorder(&csr, {&cpu_context});
  auto perm = new int[3];
  for (int i =0; i< 3; i++){
      perm[order[i]] = i;
  }
  for (int i =0; i< 2; i++){
    auto u = perm[i];
    auto v = perm[i+1];
    EXPECT_GE(xadj[v+1]-xadj[v], xadj[u+1]-xadj[u]);
  }
}
class Degrees_DegreeDistributionTest : public ::testing::Test {
protected:
    Degrees_DegreeDistribution<int, int, int, float> feature;
    sparsebase::format::CSR<int, int, int>* csr;
    sparsebase::format::COO<int, int, int>* coo;
    sparsebase::context::CPUContext cpu_context;

    void SetUp() override{
        csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr, sparsebase::format::kNotOwned);
        coo = new sparsebase::format::COO<int, int, int>(3, 3, 4, is, adj, nullptr, sparsebase::format::kNotOwned);
    }
    void TearDown() override {
        delete csr;
        delete coo;
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
    EXPECT_THROW(feature.get_params(std::type_index(typeid(int))).get(), utils::FeatureParamsException);
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
    sparsebase::format::COO<int, int, int>* coo;
    sparsebase::context::CPUContext cpu_context;

    void SetUp() override{
        csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr, sparsebase::format::kNotOwned);
        coo = new sparsebase::format::COO<int, int, int>(3, 3, 4, is, adj, nullptr, sparsebase::format::kNotOwned);
    }
    void TearDown() override {
        delete csr;
        delete coo;
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
    delete [] degrees_array;
    // Check GetDegrees with conversion
    degrees_array = feature.GetDegrees({coo}, {&cpu_context});
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
    // Check Extract with conversion
    feature_map = feature.Extract(coo, {&cpu_context});
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
    sparsebase::context::CPUContext cpu_context;

    void SetUp() override{
        csr = new sparsebase::format::CSR<int, int, int>(3, 3, xadj, adj, nullptr, sparsebase::format::kNotOwned);
        coo = new sparsebase::format::COO<int, int, int>(3, 3, 4, is, adj, nullptr, sparsebase::format::kNotOwned);
    }
    void TearDown() override {
        delete csr;
        delete coo;
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
    // Check Extract with conversion
    feature_map = feature.Extract(coo, {&cpu_context});
    // Check map size and type
    EXPECT_EQ(feature_map.size(), 1);
    for (auto feat : feature_map){
        EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
    }
    for (int i =0; i<3; i++){
        EXPECT_EQ(std::any_cast<float*>(feature_map[feature.get_feature_id()])[i], distribution[i]);
    }
}
TEST_F(Degrees_DegreeDistributionTest, Degree_DegreeDistributionTests){
    // test get_sub_ids
    EXPECT_EQ(feature.get_sub_ids().size(), 2);
    std::vector<std::type_index> ids = {Degrees<int, int, int>::get_feature_id_static(), DegreeDistribution<int, int, int, float>::get_feature_id_static()};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(feature.get_sub_ids()[0], ids[0]);
    EXPECT_EQ(feature.get_sub_ids()[1], ids[1]);

    // Test get_subs
    auto subs = feature.get_subs();
    // two sub-feature
    EXPECT_EQ(subs.size(), 2);
    // same type as feature but different address
    EXPECT_EQ(std::type_index(typeid(*(subs[0]))), ids[0]);
    EXPECT_EQ(std::type_index(typeid(*(subs[1]))), ids[1]);
    EXPECT_NE(subs[0], &feature);
    EXPECT_NE(subs[1], &feature);

    // Check GetCSR implementation function
    Params1 p1;
    auto degrees_and_distribution_map = Degrees_DegreeDistribution<int, int, int, float>::GetCSR({csr}, &p1);
    ASSERT_EQ(degrees_and_distribution_map.size(), 2);
    ASSERT_NE(degrees_and_distribution_map.find(ids[0]), degrees_and_distribution_map.end());
    ASSERT_NE(degrees_and_distribution_map.find(ids[1]), degrees_and_distribution_map.end());
    ASSERT_NO_THROW(std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
    auto distribution_array = std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
    ASSERT_NO_THROW(std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]));
    auto degree_array = std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]);
    for (int i =0; i<3; i++){
        EXPECT_EQ(distribution_array[i], distribution[i]);
        EXPECT_EQ(degree_array[i], degrees[i]);
    }
    delete [] distribution_array;
    delete [] degree_array;
    //// Check Get (function matcher)
    degrees_and_distribution_map = feature.Get({csr}, {&cpu_context});
    ASSERT_EQ(degrees_and_distribution_map.size(), 2);
    ASSERT_NE(degrees_and_distribution_map.find(ids[0]), degrees_and_distribution_map.end());
    ASSERT_NE(degrees_and_distribution_map.find(ids[1]), degrees_and_distribution_map.end());
    ASSERT_NO_THROW(std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
    distribution_array = std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
    ASSERT_NO_THROW(std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]));
    degree_array = std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]);
    for (int i =0; i<3; i++){
        EXPECT_EQ(distribution_array[i], distribution[i]);
        EXPECT_EQ(degree_array[i], degrees[i]);
    }
    delete [] distribution_array;
    delete [] degree_array;
    //// Check Get with conversion (function matcher)
    degrees_and_distribution_map = feature.Get({coo}, {&cpu_context});
    ASSERT_EQ(degrees_and_distribution_map.size(), 2);
    ASSERT_NE(degrees_and_distribution_map.find(ids[0]), degrees_and_distribution_map.end());
    ASSERT_NE(degrees_and_distribution_map.find(ids[1]), degrees_and_distribution_map.end());
    ASSERT_NO_THROW(std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
    distribution_array = std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
    ASSERT_NO_THROW(std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]));
    degree_array = std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]);
    for (int i =0; i<3; i++){
        EXPECT_EQ(distribution_array[i], distribution[i]);
        EXPECT_EQ(degree_array[i], degrees[i]);
    }
    delete [] distribution_array;
    delete [] degree_array;
    // Check Extract
    degrees_and_distribution_map = feature.Extract(csr, {&cpu_context});
    ASSERT_EQ(degrees_and_distribution_map.size(), 2);
    ASSERT_NE(degrees_and_distribution_map.find(ids[0]), degrees_and_distribution_map.end());
    ASSERT_NE(degrees_and_distribution_map.find(ids[1]), degrees_and_distribution_map.end());
    ASSERT_NO_THROW(std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
    distribution_array = std::any_cast<float*>(degrees_and_distribution_map[DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
    ASSERT_NO_THROW(std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]));
    degree_array = std::any_cast<int*>(degrees_and_distribution_map[Degrees<int, int, int>::get_feature_id_static()]);
    for (int i =0; i<3; i++){
        EXPECT_EQ(distribution_array[i], distribution[i]);
        EXPECT_EQ(degree_array[i], degrees[i]);
    }
    delete [] distribution_array;
    delete [] degree_array;
}

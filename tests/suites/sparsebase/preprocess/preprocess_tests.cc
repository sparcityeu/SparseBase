#include "sparsebase/config.h"
#include "gtest/gtest.h"

#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/converter/converter.h"
#include "sparsebase/utils/exception.h"
#include <iostream>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <vector>
using namespace sparsebase;
using namespace sparsebase::preprocess;
const int n = 3;
const int nnz = 4;
int row_ptr[n + 1] = {0, 2, 3, 4};
int cols[nnz] = {1, 2, 0, 0};
int rows[nnz] = {0, 0, 1, 2};
float distribution[n] = {2.0 / nnz, 1.0 / nnz, 1.0 / nnz};
int degrees[n] = {2, 1, 1};
int vals[nnz] = {1, 2, 3, 4};

// row reordering
// r_reorder_vector[i] = j -> new row j used to be at location i
int r_reorder_vector[3] = {1, 2, 0};
int r_row_ptr[n+1] = {0, 1, 3, 4};
int r_rows[nnz] = {0, 1, 1, 2};
int r_cols[nnz] = {0, 1, 2, 0};
int r_vals[nnz] = {4, 1, 2, 3};

// column reordering
int c_reorder_vector[3] = {2, 0, 1};
int c_row_ptr[n+1] = {0, 2, 3, 4};
int c_rows[nnz] = {0, 0, 1, 2};
int c_cols[nnz] = {0, 1, 2, 2};
int c_vals[nnz] = {1, 2, 3, 4};

// row and column reordering
int rc_row_ptr[n+1] = {0, 1, 3, 4};
int rc_rows[nnz] = {0, 1, 1, 2};
int rc_cols[nnz] = {2, 0, 1, 2};
int rc_vals[nnz] = {4, 1, 2, 3};


const int array_length = 3;
int inverse_perm_array[array_length] = {2, 0, 1};
int perm_array[array_length] = {1,2,0};
float original_array[array_length] = {0.0, 0.1, 0.2};
float reordered_array[array_length] = {0.1, 0.2, 0.0};
format::Array<float> orig_arr(n, original_array, format::kNotOwned);
format::Array<float> inv_arr(n, reordered_array, format::kNotOwned);
format::CSR<int, int, int> global_csr(n, n, row_ptr, cols, vals,
                                      format::kNotOwned);
format::COO<int, int, int> global_coo(n, n, nnz, rows, cols, vals,
                                      format::kNotOwned);
sparsebase::context::CPUContext cpu_context;

template <typename IDType>
void check_degree_ordering(IDType *order, IDType n, IDType *row_ptr,
                           bool ascending = true) {
  auto *permutation = new IDType[n];
  for (IDType i = 0; i < n; i++) {
    permutation[order[i]] = i;
  }
  bool order_is_correct = true;
  std::set<IDType> check;
  for (IDType new_u = 0; new_u < n - 1 && order_is_correct; new_u++) {
    IDType u = permutation[new_u];
    EXPECT_EQ(check.find(u), check.end());
    check.insert(u);
    IDType v = permutation[new_u + 1];
    if (ascending)
      EXPECT_LE(row_ptr[u + 1] - row_ptr[u], row_ptr[v + 1] - row_ptr[v]);
    else
      EXPECT_GE(row_ptr[u + 1] - row_ptr[u], row_ptr[v + 1] - row_ptr[v]);
  }
  IDType v = permutation[n - 1];
  EXPECT_EQ(check.find(v), check.end());
  check.insert(v);
  delete[] permutation;
}
template <typename IDType> void check_reorder(IDType *order, IDType n) {
  std::set<IDType> vertices;
  for (IDType i = 0; i < n; i++) {
    EXPECT_EQ(vertices.find(order[i]), vertices.end());
    vertices.insert(order[i]);
  }
}
template <typename IDType, typename NNZType, typename ValueType>
void compare_csr(format::CSR<IDType, NNZType, ValueType> *correct,
                 format::CSR<IDType, NNZType, ValueType> *testing) {
  auto correct_row_ptr = correct->get_row_ptr();
  auto correct_col = correct->get_col();
  auto testing_row_ptr = testing->get_row_ptr();
  auto testing_col = testing->get_col();

  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(correct_row_ptr[i], testing_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++) {
    EXPECT_EQ(correct_col[i], testing_col[i]);
  }
}
template <typename V, typename E, typename O, typename L>
void confirm_renumbered_csr(V *xadj, V *renumbered_xadj, E *adj,
                            E *renumbered_adj, O *inverse_order, L n) {
  auto order = new E[n];
  for (L i = 0; i < n; i++) {
    order[inverse_order[i]] = i;
  }
  for (L i = 0; i < n; i++) {
    EXPECT_EQ(xadj[i + 1] - xadj[i], renumbered_xadj[inverse_order[i] + 1] -
                                         renumbered_xadj[inverse_order[i]]);
    std::set<V> edges;
    for (E edge = xadj[i]; edge < xadj[i + 1]; edge++) {
      edges.insert(inverse_order[adj[edge]]);
    }
    for (E edge = renumbered_xadj[inverse_order[i]];
         edge < renumbered_xadj[inverse_order[i] + 1]; edge++) {
      EXPECT_NE(edges.find(renumbered_adj[edge]), edges.end());
    }
  }
}
TEST(ArrayPermute, Basic){
  context::CPUContext cpu_context;
  PermuteOrderOne<int, float> transform(inverse_perm_array);
  format::Format* inv_arr_fp = transform.GetTransformation(&orig_arr, {&cpu_context}, false);
  format::Array<float> *inv_arr = inv_arr_fp->As<format::Array<float>>();
  for (int i = 0; i < n; i++){
    EXPECT_EQ(inv_arr->get_vals()[i], reordered_array[i]);
  }
  EXPECT_NO_THROW(transform.GetTransformation(&orig_arr, {&cpu_context}, false));
}
TEST(ArrayPermute, Inverse){
  context::CPUContext cpu_context;
  auto inv_p = ReorderBase::InversePermutation(inverse_perm_array, global_csr.get_dimensions()[0]);
  PermuteOrderOne<int, float> inverse_transform(inv_p);
  format::Format* inv_inversed_arr_fp = inverse_transform.GetTransformation(&inv_arr, {&cpu_context}, false);
  format::Array<float> *inv_inversed_arr = inv_inversed_arr_fp->As<format::Array<float>>();
  for (int i = 0; i < n; i++){
    EXPECT_EQ(inv_inversed_arr->get_vals()[i], original_array[i]);
  }
}
TEST(TypeIndexHash, Basic) {
  TypeIndexVectorHash hasher;
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

TEST(ConverterMixin, Basics) {
  ConverterMixin<PreprocessType> instance;
  //! Check getting an empty converter
  ASSERT_EQ(instance.GetConverter(), nullptr);

  //! Check setting a converter
  utils::converter::ConverterOrderOne<int> converter;
  instance.SetConverter(converter);
  // Same type
  ASSERT_NE(dynamic_cast<utils::converter::ConverterOrderOne<int> *>(
                instance.GetConverter().get()),
            nullptr);
  // Different object
  ASSERT_NE(instance.GetConverter().get(), &converter);

  //! Check resetting a converter
  instance.ResetConverter();
  // Same type
  ASSERT_NE(dynamic_cast<utils::converter::ConverterOrderOne<int> *>(
                instance.GetConverter().get()),
            nullptr);
  // Different object
  ASSERT_NE(instance.GetConverter().get(), &converter);
}

class FunctionMatcherMixinTest : public ::testing::Test {
protected:
  GenericPreprocessType<int> concrete_preprocess;

  static int OneImplementationFunction(std::vector<format::Format *> inputs,
                                       PreprocessParams *) {
    return 1;
  }
  static int TwoImplementationFunction(std::vector<format::Format *> inputs,
                                       PreprocessParams *) {
    return 2;
  }
  static int ThreeImplementationFunction(std::vector<format::Format *> inputs,
                                         PreprocessParams *) {
    return 3;
  }
  static int FourImplementationFunction(std::vector<format::Format *> inputs,
                                        PreprocessParams *) {
    return 4;
  }
  struct GenericParams : PreprocessParams {
    GenericParams(int p) : param(p) {}
    int param;
  };
};

TEST_F(FunctionMatcherMixinTest, BlackBox) {
  format::CSR<int, int, int> *csr = &global_csr;
  // Check calling with an empty map
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
               utils::FunctionNotFoundException);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
               utils::FunctionNotFoundException);

  // Check calling with no conversion needed
  concrete_preprocess.RegisterFunction({csr->get_format_id()},
                                       OneImplementationFunction);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true), 1);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false), 1);

  // Check unregistering
  EXPECT_EQ(
      concrete_preprocess.UnregisterFunction(
          {sparsebase::format::CSR<int, int, int>::get_format_id_static()}),
      true);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
               utils::FunctionNotFoundException);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
               utils::FunctionNotFoundException);

  // Check unregistering an already unregistered key
  EXPECT_EQ(
      concrete_preprocess.UnregisterFunction(
          {sparsebase::format::CSR<int, int, int>::get_format_id_static()}),
      false);

  // Check calling with one conversion needed and no converter registered
  concrete_preprocess.RegisterFunction(
      {sparsebase::format::COO<int, int, int>::get_format_id_static()},
      TwoImplementationFunction);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true),
               utils::NoConverterException);
  // should fail with different exception
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false),
               utils::NoConverterException);

  // Check calling with one conversion needed and a converter registered
  concrete_preprocess.SetConverter(
      utils::converter::ConverterOrderTwo<int, int, int>{});
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true), 2);
  EXPECT_THROW(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);

  // Check calling with no conversion needed even though one is possible
  concrete_preprocess.RegisterFunction({csr->get_format_id()},
                                       OneImplementationFunction);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true), 1);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false), 1);

  // Checking override
  // Override an existing function in the map
  concrete_preprocess.RegisterFunction({csr->get_format_id()},
                                       ThreeImplementationFunction);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true), 3);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false), 3);

  // Try to override but fail
  EXPECT_EQ(concrete_preprocess.RegisterFunctionNoOverride(
                {csr->get_format_id()}, FourImplementationFunction),
            false);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true), 3);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false), 3);

  // Try to override and succeed
  concrete_preprocess.UnregisterFunction(
      {sparsebase::format::CSR<int, int, int>::get_format_id_static()});
  EXPECT_EQ(concrete_preprocess.RegisterFunctionNoOverride(
                {csr->get_format_id()}, FourImplementationFunction),
            true);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, true), 4);
  EXPECT_EQ(concrete_preprocess.GetOutput(csr, nullptr, {&cpu_context}, false), 4);

  // Checking cached getters
  // No conversion needed to be done
  auto tup = concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, true);
  EXPECT_EQ(std::get<0>(tup)[0], nullptr);
  EXPECT_EQ(std::get<1>(tup), 4);
  tup = concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, false);
  EXPECT_EQ(std::get<0>(tup)[0], nullptr);
  EXPECT_EQ(std::get<1>(tup), 4);

  // One conversion is done
  concrete_preprocess.UnregisterFunction(
      {sparsebase::format::CSR<int, int, int>::get_format_id_static()});
  auto tup2 = concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, true);
  ASSERT_NE(std::get<0>(tup2)[0], nullptr);
  ASSERT_NE(std::get<0>(tup2)[0]->get_format_id(), csr->get_format_id());
  EXPECT_EQ(std::get<1>(tup2), 2);
  EXPECT_THROW(concrete_preprocess.GetOutputCached(csr, nullptr, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
}


TEST(DegreeReorder, AscendingOrder) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(true);
  auto order = reorder.GetReorder(&global_csr, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr);
}
TEST(DegreeReorder, DescendingOrder) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_csr, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
}
void CompareVectorsOfTypeIndex(std::vector<std::type_index> i1, std::vector<std::type_index> i2){
  ASSERT_EQ(i1.size(), i2.size());
  std::sort(i1.begin(), i1.end());
  std::sort(i2.begin(), i2.end());
  for (int i =0; i < i1.size(); i++){
    EXPECT_EQ(i1[i], i2[i]);
  }
}
TEST(DegreeReorder, TwoParamsConversion) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  EXPECT_THROW(reorder.GetReorder(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  try {
    reorder.GetReorder(&global_coo, {&cpu_context}, true);
  } catch(utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>& exception){
    CompareVectorsOfTypeIndex(exception.used_format_, {format::CSR<int,int,int>::get_format_id_static()});
    auto class_available_formats = reorder.GetAvailableFormats();
    auto returned_available_formats = exception.available_formats_;
    sort(class_available_formats.begin(), class_available_formats.end());
    sort(returned_available_formats.begin(), returned_available_formats.end());
    for (int i =0 ; i< class_available_formats.size(); i++){
      CompareVectorsOfTypeIndex(class_available_formats[i], returned_available_formats[i]);
    }
  }
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
}
TEST(ReorderTypeTest, DescendingWithParams) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(true);
  sparsebase::preprocess::DegreeReorder<int, int, int>::DegreeReorderParams
      param(false);
  auto order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
  EXPECT_NO_THROW(reorder.GetReorder(&global_csr, &param, {&cpu_context}, true));
  order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, false);
}
TEST(ReorderTypeTest, AscendingWithParams) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  sparsebase::preprocess::DegreeReorder<int, int, int>::DegreeReorderParams
      param(true);
  auto order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
  EXPECT_NO_THROW(reorder.GetReorder(&global_csr, &param, {&cpu_context}, true));
  order = reorder.GetReorder(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
}
TEST(ReorderTypeTest, NoCachConversion) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  sparsebase::preprocess::DegreeReorder<int, int, int>::DegreeReorderParams
      param(true);
  auto order = reorder.GetReorder(&global_coo, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
  EXPECT_NO_THROW(reorder.GetReorder(&global_coo, &param, {&cpu_context}, true));
  order = reorder.GetReorder(&global_coo, &param, {&cpu_context}, true);
  check_degree_ordering(order, n, row_ptr, true);
}

TEST(ReorderTypeTest, CachedNoConversion) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  sparsebase::preprocess::DegreeReorder<int, int, int>::DegreeReorderParams
      param(true);
  auto order = reorder.GetReorderCached(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, true);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_EQ(std::get<0>(order)[0], nullptr);
  EXPECT_NO_THROW(reorder.GetReorderCached(&global_csr, &param, {&cpu_context}, true));
  order = reorder.GetReorderCached(&global_csr, &param, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, true);
}

TEST(ReorderTypeTest, CachedConversionTwoParams) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorderCached(&global_coo, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, false);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_NE(std::get<0>(order)[0], nullptr);
  auto cached_csr = std::get<0>(order)[0]->As<format::CSR<int, int, int>>();
  compare_csr(&global_csr, cached_csr);
  EXPECT_THROW(reorder.GetReorderCached(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  try {
    reorder.GetReorderCached(&global_coo, {&cpu_context}, true);
  } catch(utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>& exception){
    CompareVectorsOfTypeIndex(exception.used_format_, {format::CSR<int,int,int>::get_format_id_static()});
    auto class_available_formats = reorder.GetAvailableFormats();
    auto returned_available_formats = exception.available_formats_;
    sort(class_available_formats.begin(), class_available_formats.end());
    sort(returned_available_formats.begin(), returned_available_formats.end());
    for (int i =0 ; i< class_available_formats.size(); i++){
      CompareVectorsOfTypeIndex(class_available_formats[i], returned_available_formats[i]);
    }
  }
}

TEST(ReorderTypeTest, CachedNoConversionTwoParams) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorderCached(&global_csr, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, false);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_EQ(std::get<0>(order)[0], nullptr);
  EXPECT_NO_THROW(reorder.GetReorderCached(&global_csr, {&cpu_context}, true));
  order = reorder.GetReorderCached(&global_csr, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, false);
}

TEST(ReorderTypeTest, CachedConversion) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  sparsebase::preprocess::DegreeReorder<int, int, int>::DegreeReorderParams
      param(true);
  auto order = reorder.GetReorderCached(&global_coo, &param, {&cpu_context}, true);
  check_degree_ordering(std::get<1>(order), n, row_ptr, true);
  EXPECT_EQ(std::get<0>(order).size(), 1);
  EXPECT_NE(std::get<0>(order)[0], nullptr);
  auto cached_csr = std::get<0>(order)[0]->As<format::CSR<int, int, int>>();
  compare_csr(&global_csr, cached_csr);
  EXPECT_THROW(reorder.GetReorderCached(&global_coo, &param, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  try {
    reorder.GetReorderCached(&global_coo, &param, {&cpu_context}, true);
  } catch(utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>& exception){
    CompareVectorsOfTypeIndex(exception.used_format_, {format::CSR<int,int,int>::get_format_id_static()});
    auto class_available_formats = reorder.GetAvailableFormats();
    auto returned_available_formats = exception.available_formats_;
    sort(class_available_formats.begin(), class_available_formats.end());
    sort(returned_available_formats.begin(), returned_available_formats.end());
    for (int i =0 ; i< class_available_formats.size(); i++){
      CompareVectorsOfTypeIndex(class_available_formats[i], returned_available_formats[i]);
    }
  }
}

TEST(RCMReorderTest, BasicTest) {
  sparsebase::preprocess::RCMReorder<int, int, int> reorder;
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  check_reorder(order, n);
}
template <typename a, typename b, typename c>
class TestFormat : public sparsebase::format::FormatImplementation<TestFormat<a,b,c>, sparsebase::format::FormatOrderTwo<a,b,c>>{
  format::Format *Clone() const override{
    return nullptr;
  }

};

template <typename a>
class TestFormatOrderOne : public sparsebase::format::FormatImplementation<TestFormatOrderOne<a>, sparsebase::format::FormatOrderOne<a>>{
  format::Format *Clone() const override{
    return nullptr;
  }

};
TEST(ReorderBase, Reorder) {
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Reorder<RCMReorder>({}, &global_csr, {&cpu_context}, true));
  auto order = sparsebase::preprocess::ReorderBase::Reorder<RCMReorder>({}, &global_csr, {&cpu_context}, true);
  check_reorder(order, n);
}

TEST(ReorderBase, Permute1D) {
  // no conversion of output
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute1D(inverse_perm_array, &orig_arr, {&cpu_context}, true));
  // check output of permutation
  format::Format* inv_arr_fp = sparsebase::preprocess::ReorderBase::Permute1D(inverse_perm_array, &orig_arr, {&cpu_context}, true);

  auto *inv_arr = inv_arr_fp->As<format::Array<float>>();
  for (int i = 0; i < n; i++){
    EXPECT_EQ(inv_arr->get_vals()[i], reordered_array[i]);
  }
  // converting output to possible format
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute1D<format::Array>(inverse_perm_array, &orig_arr, {&cpu_context}, true));
  EXPECT_EQ((sparsebase::preprocess::ReorderBase::Permute1D<format::Array>(inverse_perm_array, &orig_arr, {&cpu_context}, true))->get_format_id(), (format::Array<float>::get_format_id_static()));
  inv_arr = sparsebase::preprocess::ReorderBase::Permute1D<format::Array>(inverse_perm_array, &orig_arr, {&cpu_context}, true);

  for (int i = 0; i < n; i++){
    EXPECT_EQ(inv_arr->get_vals()[i], reordered_array[i]);
  }
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &global_csr, {&cpu_context}, false), utils::ConversionException);
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute1D<TestFormatOrderOne>(inverse_perm_array, &orig_arr, {&cpu_context}, true), utils::ConversionException);
  // passing a format that isn't convertable
  TestFormatOrderOne<int> f;
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute1D(r_reorder_vector, &f, {&cpu_context}, true), utils::FunctionNotFoundException);
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute1D(r_reorder_vector, &f, {&cpu_context}, false), utils::FunctionNotFoundException);
}
TEST(ReorderBase, Permute2D) {
  // no conversion of output
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2D(r_reorder_vector, &global_csr, {&cpu_context}, true));
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2D(r_reorder_vector, &global_csr, {&cpu_context}, false));
  // check output of permutation
  auto transformed_format = sparsebase::preprocess::ReorderBase::Permute2D(r_reorder_vector, &global_csr, {&cpu_context}, true)->Convert<format::CSR>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), r_reorder_vector, n);
  // converting output to possible format
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2D<format::CSR>(r_reorder_vector, &global_csr, {&cpu_context}, false));
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2D<format::CSR>(r_reorder_vector, &global_csr, {&cpu_context}, true));
  EXPECT_EQ((sparsebase::preprocess::ReorderBase::Permute2D<format::CSR>(r_reorder_vector, &global_csr, {&cpu_context}, true))->get_format_id(), (format::CSR<int, int, int>::get_format_id_static()));
  transformed_format = sparsebase::preprocess::ReorderBase::Permute2D<format::CSR>(r_reorder_vector, &global_csr, {&cpu_context}, true);
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), r_reorder_vector, n);
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &global_csr, {&cpu_context}, true), utils::ConversionException);
  // passing a format that isn't convertable
  TestFormat<int, int, int> f;
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2D<TestFormat>(r_reorder_vector, &f, {&cpu_context}, true), utils::FunctionNotFoundException);
}
TEST(ReorderBase, InversePermutation) {
  auto perm = preprocess::ReorderBase::InversePermutation(inverse_perm_array, orig_arr.get_dimensions()[0]);
  for (int i =0; i < array_length; i++){
    EXPECT_EQ(perm[i], perm_array[i]);
  }
}
TEST(ReorderBase, Permute2DRowColWise) {
  // no conversion of output
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true));
  // check output of permutation
  auto transformed_format = sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true)->Convert<format::CSR>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], rc_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], rc_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], rc_vals[i]);
  }
  // converting output to possible format
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true));
  EXPECT_EQ((sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true))->get_format_id(), (format::CSR<int, int, int>::get_format_id_static()));
  transformed_format = sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise(r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true)->Convert<format::CSR>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], rc_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], rc_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], rc_vals[i]);
  }
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise<TestFormat>(r_reorder_vector, c_reorder_vector, &global_csr, {&cpu_context}, true), utils::ConversionException);
  // passing a format that isn't convertable
  TestFormat<int, int, int> f;
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise<TestFormat>(r_reorder_vector, c_reorder_vector, &f, {&cpu_context}, true), utils::FunctionNotFoundException);
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowColumnWise<TestFormat>(r_reorder_vector, c_reorder_vector, &f, {&cpu_context}, false), utils::FunctionNotFoundException);
}
TEST(ReorderBase, Permute2DRowWise) {
  // no conversion of output
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr, {&cpu_context}, true));
  // check output of permutation
  auto transformed_format = sparsebase::preprocess::ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr, {&cpu_context}, true)->Convert<format::CSR>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], r_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], r_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], r_vals[i]);
  }
  // converting output to possible format
  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr, {&cpu_context}, true));
  EXPECT_EQ((sparsebase::preprocess::ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr, {&cpu_context}, true))->get_format_id(), (format::CSR<int, int, int>::get_format_id_static()));
  transformed_format = sparsebase::preprocess::ReorderBase::Permute2DRowWise(r_reorder_vector, &global_csr, {&cpu_context}, true)->Convert<format::CSR>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], r_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], r_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], r_vals[i]);
  }
  // converting output to illegal format (No conversion available)
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowWise<TestFormat>(r_reorder_vector, &global_csr, {&cpu_context}, true), utils::ConversionException);
  // passing a format that isn't convertable
  TestFormat<int, int, int> f;
  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DRowWise<TestFormat>(r_reorder_vector, &f, {&cpu_context}, true), utils::FunctionNotFoundException);
}

//TEST(ReorderBase, Permute2DColWise) {
//  // no conversion of output
//  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2DColWise(c_reorder_vector, &global_csr, {&cpu_context}));
//  // check output of permutation
//  auto transformed_format = sparsebase::preprocess::ReorderBase::Permute2DColWise(c_reorder_vector, &global_csr, {&cpu_context})->Convert<format::CSR>();
//  for (int i = 0; i < n+1; i++){
//    EXPECT_EQ(transformed_format->get_row_ptr()[i], c_row_ptr[i]);
//  }
//  for (int i = 0; i < nnz; i++){
//    EXPECT_EQ(transformed_format->get_col()[i], c_cols[i]);
//    EXPECT_EQ(transformed_format->get_vals()[i], c_vals[i]);
//  }
//  // converting output to possible format
//  EXPECT_NO_THROW(sparsebase::preprocess::ReorderBase::Permute2DColWise(c_reorder_vector, &global_csr, {&cpu_context}));
//  EXPECT_EQ((sparsebase::preprocess::ReorderBase::Permute2DColWise(c_reorder_vector, &global_csr, {&cpu_context}))->get_format_id(), (format::CSR<int, int, int>::get_format_id_static()));
//  transformed_format = sparsebase::preprocess::ReorderBase::Permute2DColWise(c_reorder_vector, &global_csr, {&cpu_context})->Convert<format::CSR>();
//  for (int i = 0; i < n+1; i++){
//    EXPECT_EQ(transformed_format->get_row_ptr()[i], c_row_ptr[i]);
//  }
//  for (int i = 0; i < nnz; i++){
//    EXPECT_EQ(transformed_format->get_col()[i], c_cols[i]);
//    EXPECT_EQ(transformed_format->get_vals()[i], c_vals[i]);
//  }
//  // converting output to illegal format (No conversion available)
//  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DColWise<TestFormat>(c_reorder_vector, &global_csr, {&cpu_context}), utils::ConversionException);
//  // passing a format that isn't convertable
//  TestFormat<int, int, int> f;
//  EXPECT_THROW(sparsebase::preprocess::ReorderBase::Permute2DColWise<TestFormat>(c_reorder_vector, &f, {&cpu_context}), utils::FunctionNotFoundException);
//}

TEST(PermuteTest, RowWise) {
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(r_reorder_vector, nullptr);
  EXPECT_THROW(transformer.GetTransformation(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  auto transformed_format =
      transformer.GetTransformation(&global_coo, {&cpu_context}, true)
          ->As<format::CSR<int, int, int>>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], r_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], r_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], r_vals[i]);
  }
}

TEST(InversePermuteTest, RowColWise) {
  sparsebase::format::CSR<int, int, int> rc_reordered_csr(3, 3, rc_row_ptr, rc_cols, rc_vals, sparsebase::format::kNotOwned, true);
  auto inv_r_order = sparsebase::preprocess::ReorderBase::InversePermutation(r_reorder_vector, rc_reordered_csr.get_dimensions()[0]);
  auto inv_c_order = sparsebase::preprocess::ReorderBase::InversePermutation(c_reorder_vector, rc_reordered_csr.get_dimensions()[0]);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(inv_r_order, inv_c_order);
  auto transformed_format =
      transformer.GetTransformation(&rc_reordered_csr, {&cpu_context}, true)
          ->As<format::CSR<int, int, int>>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], vals[i]);
  }
}

//TEST(PermuteTest, ColWise) {
//  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(nullptr, c_reorder_vector);
//  auto transformed_format =
//      transformer.GetTransformation(&global_coo, {&cpu_context})
//          ->As<format::CSR<int, int, int>>();
//  for (int i = 0; i < n+1; i++){
//    EXPECT_EQ(transformed_format->get_row_ptr()[i], c_row_ptr[i]);
//  }
//  for (int i = 0; i < nnz; i++){
//    EXPECT_EQ(transformed_format->get_col()[i], c_cols[i]);
//    EXPECT_EQ(transformed_format->get_vals()[i], c_vals[i]);
//  }
//}

TEST(PermuteTest, RowColWise) {
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(r_reorder_vector, c_reorder_vector);
  auto transformed_format =
      transformer.GetTransformation(&global_coo, {&cpu_context}, true)
          ->As<format::CSR<int, int, int>>();
  for (int i = 0; i < n+1; i++){
    EXPECT_EQ(transformed_format->get_row_ptr()[i], rc_row_ptr[i]);
  }
  for (int i = 0; i < nnz; i++){
    EXPECT_EQ(transformed_format->get_col()[i], rc_cols[i]);
    EXPECT_EQ(transformed_format->get_vals()[i], rc_vals[i]);
  }
}

TEST(PermuteTest, ConversionNoParam) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(order, order);
  auto transformed_format =
      transformer.GetTransformation(&global_coo, {&cpu_context}, true)
          ->As<format::CSR<int, int, int>>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), order, n);
  EXPECT_THROW(transformer.GetTransformation(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
}

TEST(PermuteTest, WrongInputType) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(order, order);
  EXPECT_THROW((transformer.GetTransformation(&orig_arr, {&cpu_context}, true)
                   ->As<format::CSR<int, int, int>>()), utils::TypeException);
}

TEST(PermuteTest, NoConversionParam) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_csr, {&cpu_context}, true);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(nullptr, nullptr);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int>::PermuteOrderTwoParams params(
      order, order);
  auto transformed_format =
      transformer.GetTransformation(&global_csr, &params, {&cpu_context}, true)
          ->As<format::CSR<int, int, int>>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), order, n);
  EXPECT_NO_THROW((transformer.GetTransformation(&global_csr, &params, {&cpu_context}, true)
          ->As<format::CSR<int, int, int>>()));
}

TEST(PermuteTest, ConversionParamCached) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(nullptr, nullptr);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int>::PermuteOrderTwoParams params(
      order, order);
  auto transformed_output =
      transformer.GetTransformationCached(&global_coo, &params, {&cpu_context}, true);
  auto transformed_format =
      std::get<1>(transformed_output)->As<format::CSR<int, int, int>>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), order, n);
  EXPECT_EQ(std::get<0>(transformed_output).size(), 1);
  ASSERT_NE(std::get<0>(transformed_output)[0], nullptr);
  auto cached_format =
      std::get<0>(transformed_output)[0]->As<format::CSR<int, int, int>>();
  compare_csr(&global_csr, cached_format);
}

TEST(PermuteTest, NoConversionNoParamCached) {
  sparsebase::preprocess::DegreeReorder<int, int, int> reorder(false);
  auto order = reorder.GetReorder(&global_coo, {&cpu_context}, true);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int> transformer(nullptr, nullptr);
  sparsebase::preprocess::PermuteOrderTwo<int, int, int>::PermuteOrderTwoParams params(
      order, order);
  auto transformed_output =
      transformer.GetTransformationCached(&global_csr, &params, {&cpu_context}, true);
  auto transformed_format =
      std::get<1>(transformed_output)->As<format::CSR<int, int, int>>();
  confirm_renumbered_csr(
      global_csr.get_row_ptr(), transformed_format->get_row_ptr(),
      global_csr.get_col(), transformed_format->get_col(), order, n);
  EXPECT_EQ(std::get<0>(transformed_output).size(), 1);
  ASSERT_EQ(std::get<0>(transformed_output)[0], nullptr);
}

#ifndef USE_CUDA
TEST(JaccardTest, NoCuda) {
  sparsebase::preprocess::JaccardWeights<int, int, int, float> jac;
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, true),
               utils::FunctionNotFoundException);
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, false),
               utils::FunctionNotFoundException);
}
#endif

class Degrees_DegreeDistributionTest : public ::testing::Test {
protected:
  Degrees_DegreeDistribution<int, int, int, float> feature;

  struct Params1 : sparsebase::preprocess::PreprocessParams {};
  struct Params2 : sparsebase::preprocess::PreprocessParams {};
};

TEST_F(Degrees_DegreeDistributionTest, FeaturePreprocessTypeTests) {
  std::shared_ptr<Params1> p1(new Params1);
  std::shared_ptr<Params2> p2(new Params2);
  // Check getting feature id
  EXPECT_EQ(
      std::type_index(typeid(Degrees_DegreeDistribution<int, int, int, float>)),
      feature.get_feature_id());
  // Getting a params object for an unset params
  EXPECT_THROW(feature.get_params(std::type_index(typeid(int))).get(),
               utils::FeatureParamsException);
  // Checking setting params of a sub-feature
  feature.set_params(Degrees<int, int, int>::get_feature_id_static(), p1);
  EXPECT_EQ(
      feature.get_params(Degrees<int, int, int>::get_feature_id_static()).get(),
      p1.get());
  EXPECT_NE(
      feature.get_params(Degrees<int, int, int>::get_feature_id_static()).get(),
      p2.get());
  // Checking setting params of feature that isn't a sub-feature
  EXPECT_THROW(feature.set_params(typeid(int), p1),
               utils::FeatureParamsException);
}
class DegreesTest : public ::testing::Test {
protected:
  Degrees<int, int, int> feature;

  struct Params1 : sparsebase::preprocess::PreprocessParams {};
  struct Params2 : sparsebase::preprocess::PreprocessParams {};
};

TEST_F(DegreesTest, AllTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 1);
  EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

  // Test get_subs
  auto subs = feature.get_subs();
  // a single sub-feature
  EXPECT_EQ(subs.size(), 1);
  // same type as feature but different address
  auto& feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)),
            std::type_index(typeid(feature)));
  EXPECT_NE(subs[0], &feature);

  // Check GetDegreesCSR implementation function
  Params1 p1;
  auto degrees_array =
      Degrees<int, int, int>::GetDegreesCSR({&global_csr}, &p1);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  delete[] degrees_array;
  // Check GetDegrees
  degrees_array = feature.GetDegrees(&global_csr, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  delete[] degrees_array;
  degrees_array = feature.GetDegrees(&global_csr, {&cpu_context}, false);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  delete[] degrees_array;
  // Check GetDegrees with conversion
  degrees_array = feature.GetDegrees(&global_coo, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  EXPECT_THROW(feature.GetDegrees(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  // Check Extract
  auto feature_map = feature.Extract(&global_csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<int *>(feature_map[feature.get_feature_id()])[i],
              degrees[i]);
  }
  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<int *>(feature_map[feature.get_feature_id()])[i],
              degrees[i]);
  }
  EXPECT_THROW(feature.Extract(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
}
class DegreeDistributionTest : public ::testing::Test {
protected:
  DegreeDistribution<int, int, int, float> feature;

  struct Params1 : sparsebase::preprocess::PreprocessParams {};
  struct Params2 : sparsebase::preprocess::PreprocessParams {};
};

TEST_F(DegreeDistributionTest, AllTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 1);
  EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

  // Test get_subs
  auto subs = feature.get_subs();
  // a single sub-feature
  EXPECT_EQ(subs.size(), 1);
  // same type as feature but different address
  auto& feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)),
            std::type_index(typeid(feature)));
  EXPECT_NE(subs[0], &feature);

  // Check GetDegreeDistributionCSR implementation function
  Params1 p1;
  auto distribution_array =
      DegreeDistribution<int, int, int, float>::GetDegreeDistributionCSR(
          {&global_csr}, &p1);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  //// Check GetDistribution (function matcher)
  distribution_array = feature.GetDistribution(&global_csr, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  distribution_array = feature.GetDistribution(&global_csr, {&cpu_context}, false);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  // Check GetDistribution with conversion
  distribution_array = feature.GetDistribution(&global_coo, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  EXPECT_THROW(feature.GetDistribution(&global_coo, {&cpu_context}, false),utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  // Check GetDistribution with conversion and cached
  auto distribution_array_format =
      feature.GetDistributionCached(&global_coo, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::get<1>(distribution_array_format)[i], distribution[i]);
  }
  delete[] std::get<1>(distribution_array_format);
  auto cached_data = std::get<0>(distribution_array_format);
  ASSERT_EQ(cached_data.size(), 1);
  ASSERT_EQ(cached_data[0]->get_format_id(),
            std::type_index(typeid(global_csr)));
  auto converted_csr = cached_data[0]->As<format::CSR<int, int, int>>();
  compare_csr(&global_csr, converted_csr);
  // Check Extract
  auto feature_map = feature.Extract(&global_csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<float *>(feature_map[feature.get_feature_id()])[i],
              distribution[i]);
  }
  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<float *>(feature_map[feature.get_feature_id()])[i],
              distribution[i]);
  }
}
TEST_F(Degrees_DegreeDistributionTest, Degree_DegreeDistributionTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 2);
  std::vector<std::type_index> ids = {
      Degrees<int, int, int>::get_feature_id_static(),
      DegreeDistribution<int, int, int, float>::get_feature_id_static()};
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(feature.get_sub_ids()[0], ids[0]);
  EXPECT_EQ(feature.get_sub_ids()[1], ids[1]);

  // Test get_subs
  auto subs = feature.get_subs();
  // two sub-feature
  EXPECT_EQ(subs.size(), 2);
  // same type as feature but different address
  auto& feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), ids[0]);
  auto& feat1 = *(subs[1]);
  EXPECT_EQ(std::type_index(typeid(feat1)), ids[1]);
  EXPECT_NE(subs[0], &feature);
  EXPECT_NE(subs[1], &feature);

  // Check GetCSR implementation function
  Params1 p1;
  auto degrees_and_distribution_map =
      Degrees_DegreeDistribution<int, int, int, float>::GetCSR({&global_csr},
                                                               &p1);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
  auto distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]));
  auto degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
  //// Check Get (function matcher)
  degrees_and_distribution_map = feature.Get(&global_csr, {&cpu_context}, true);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
  distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]));
  degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
  //// Check Get with conversion (function matcher)
  degrees_and_distribution_map = feature.Get(&global_coo, {&cpu_context}, true);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
  distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]));
  degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
  EXPECT_THROW(feature.Get(&global_coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  // Check Extract
  degrees_and_distribution_map = feature.Extract(&global_csr, {&cpu_context}, true);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]));
  distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_feature_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]));
  degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_feature_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
}

TEST(GraphFeatureBase, Degrees){
 EXPECT_NO_THROW(sparsebase::preprocess::GraphFeatureBase::GetDegrees({}, &global_csr, {&cpu_context}, true));
 auto degrees_array = sparsebase::preprocess::GraphFeatureBase::GetDegrees({}, &global_csr, {&cpu_context}, true);
 EXPECT_EQ(std::type_index(typeid(degrees_array)), std::type_index(typeid(int*)));
 for (int i = 0; i < n; i++) {
   EXPECT_EQ(degrees_array[i], degrees[i]);
 }
}

TEST(GraphFeatureBase, DegreeDistribution){
  EXPECT_NO_THROW(sparsebase::preprocess::GraphFeatureBase::GetDegreeDistribution<float>({}, &global_csr, {&cpu_context}, true));
  auto degreeDistribution_array = sparsebase::preprocess::GraphFeatureBase::GetDegreeDistribution<float>({}, &global_csr, {&cpu_context}, true);
  EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)), std::type_index(typeid(float*)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
  }
}

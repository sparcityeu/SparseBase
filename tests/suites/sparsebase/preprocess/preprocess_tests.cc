#include "gtest/gtest.h"
#include "sparsebase/config.h"

#include <iostream>
#include "sparsebase/format/format.h"
#include "sparsebase/context/context.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/converter/converter.h"
#include <typeindex>
#include <typeinfo>
#include <vector>
TEST(TypeIndexHash, Basic){
  sparsebase::preprocess::TypeIndexVectorHash hasher;
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
  sparsebase::preprocess::ConverterMixin<sparsebase::preprocess::PreprocessType> instance;
  sparsebase::utils::converter::ConverterOrderOne<int> converter;
  // Check setting a converter
  ASSERT_EQ(instance.GetConverter(), nullptr);
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
    std::cout << u << " " << v << std::endl;
    EXPECT_GE(xadj[v+1]-xadj[v], xadj[u+1]-xadj[u]);
  }
}
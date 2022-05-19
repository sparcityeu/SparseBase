#include "gtest/gtest.h"
#include "sparsebase/config.h"

#include <iostream>
#include "sparsebase/format/format.h"
#include "sparsebase/context/context.h"
#include "sparsebase/preprocess/preprocess.h"
TEST(DegreeReorder, AscendingOrder){
  int xadj[4] = {0, 2, 3, 4};
  int adj[4] = {1,2,0,0};
  sparsebase::context::CPUContext cpu_context;
  sparsebase::format::CSR<int, int, int> csr(3, 3, xadj, adj, nullptr);
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
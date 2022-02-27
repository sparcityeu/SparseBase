
#include "gtest/gtest.h"

#include "sparsebase/sparse_format.h"

TEST(SparseFormat, CreationTest){
  unsigned int xadj[4] = {0, 2, 3, 4};
  unsigned int adj[4] = {1,2,0,0};
  sparsebase::format::CSR<unsigned int, unsigned int, unsigned int> csr(3, 3, xadj, adj, nullptr);
  EXPECT_EQ(csr.get_order(), 2);
}
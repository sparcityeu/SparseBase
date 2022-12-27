#include <unordered_set>

#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"

TEST(MTXWriter, COO) {
  // Initialize a COO for testing
  int row[4]{1, 2, 3, 4};
  int col[4]{5, 6, 7, 8};
  float vals[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writer("writer_test_coo_mtx.mtx");
  writer.WriteCOO(&coo);

  // Read the COO from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> reader("reader_test_coo_mtx.mtx");
  auto coo2 = reader.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo.get_dimensions(), coo2->get_dimensions());
  EXPECT_EQ(coo.get_num_nnz(), coo2->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo.get_row()[i], coo2->get_row()[i]);
    EXPECT_EQ(coo.get_col()[i], coo2->get_col()[i]);
    EXPECT_EQ(coo.get_vals()[i], coo2->get_vals()[i]);
  }
  //TODO: add more tests
}

TEST(MTXWriter, CSR) {
  //TODO: write tests for CSR
}
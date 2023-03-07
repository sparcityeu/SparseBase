#include <unordered_set>
#include <string>

#include "gtest/gtest.h"
//#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
#include "sparsebase/io/mtx_writer.h"

TEST(MTXWriter, WriteCOO) {
  // Initialize a COO for testing
  int row_1[4]{1, 2, 3, 4};
  int col_1[4]{5, 6, 7, 8};
  float vals_1[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo_1(4, 4, 4, row_1, col_1, vals_1,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCOO_1("writer_test_coo_mtx.mtx");
  writerCOO_1.WriteCOO(&coo_1);

  // Read the COO from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> readerCOO_1("writer_test_coo_mtx.mtx");

  auto coo_1_r = readerCOO_1.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_1.get_dimensions(), coo_1_r->get_dimensions());
  EXPECT_EQ(coo_1.get_num_nnz(), coo_1_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo_1.get_row()[i], coo_1_r->get_row()[i]);
    EXPECT_EQ(coo_1.get_col()[i], coo_1_r->get_col()[i]);
    EXPECT_EQ(coo_1.get_vals()[i], coo_1_r->get_vals()[i]);
  }
  //TODO: add more tests
}

TEST(MTXWriter, WriteCSR) {
  // Initialize a CSR for testing
  int row_1[4]{1, 2, 3, 4};
  int col_1[4]{5, 6, 7, 8};
  float vals_1[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::CSR<int, int, float> csr_1(4, 4, row_1, col_1, vals_1,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCSR_1("writer_test_csr_mtx.mtx");
  writerCSR_1.WriteCSR(&csr_1);

  // Read the COO from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> readerCSR_1("writer_test_csr_mtx.mtx");

  auto csr_1_r = readerCSR_1.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr_1.get_dimensions(), csr_1_r->get_dimensions());
  EXPECT_EQ(csr_1.get_num_nnz(), csr_1_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr_1.get_row_ptr()[i], csr_1_r->get_row_ptr()[i]);
    EXPECT_EQ(csr_1.get_col()[i], csr_1_r->get_col()[i]);
    EXPECT_EQ(csr_1.get_vals()[i], csr_1_r->get_vals()[i]);
  }
}
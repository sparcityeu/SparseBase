#include <unordered_set>
#include <string>

#include "gtest/gtest.h"
//#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
#include "sparsebase/io/mtx_writer.h"

TEST(MTXWriter, WriteCOO) {
  // Initialize a COO for testing
  int row_1[4]{0, 1, 2, 3};
  int col_1[4]{0, 2, 1, 3};
  float vals_1[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo_1(4, 4, 4, row_1, col_1, vals_1,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCOO_1("writer_test_coo_mtx1.mtx");
  writerCOO_1.WriteCOO(&coo_1);

  // Read the COO from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> readerCOO_1("writer_test_coo_mtx1.mtx");

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
}

TEST(MTXWriter, WriteCOO_falseSymmetric) {
  // Initialize a COO for testing
  int row_2[4]{0, 1, 3, 3};
  int col_2[4]{0, 2, 1, 3};
  float vals_2[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo_2(4, 4, 4, row_2, col_2, vals_2,
                                               sparsebase::format::kNotOwned);
  
  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCOO_2("writer_test_coo_mtx2.mtx", "matrix", "coordinate", "real", "symmetric");
  EXPECT_THROW(
      (writerCOO_2.WriteCOO(&coo_2)),
      sparsebase::utils::ReaderException);
}

TEST(MTXWriter, WriteCOO_trueSymmetric) {
  // Initialize a COO for testing
  int row_3[3]{0, 1, 2};
  int col_3[3]{0, 2, 1};
  float vals_3[3]{0.1, 0.3, 0.3};
  sparsebase::format::COO<int, int, float> coo_3(4, 4, 3, row_3, col_3, vals_3,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCOO_3("writer_test_coo_mtx3.mtx", "matrix", "coordinate", "real", "symmetric");
  writerCOO_3.WriteCOO(&coo_3);

  // Read the COO from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> readerCOO_3("writer_test_coo_mtx3.mtx");

  auto coo_3_r = readerCOO_3.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_3.get_dimensions(), coo_3_r->get_dimensions());
  EXPECT_EQ(coo_3.get_num_nnz(), coo_3_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(coo_3.get_row()[i], coo_3_r->get_row()[i]);
    EXPECT_EQ(coo_3.get_col()[i], coo_3_r->get_col()[i]);
    EXPECT_EQ(coo_3.get_vals()[i], coo_3_r->get_vals()[i]);
  }
}

TEST(MTXWriter, WriteCOO_array) {
  // Initialize a COO for testing
  int row_4[5]{1, 2, 3, 4, 4};
  int col_4[5]{0, 1, 0, 2, 3};
  float vals_4[5]{0.1, 0.3, 0.2, 0.4, 0.5};

  sparsebase::format::COO<int, int, float> coo_4(5, 5, 5, row_4, col_4, vals_4,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCOO_4("writer_test_coo_mtx4.mtx", "matrix", "array", "real", "general");
  writerCOO_4.WriteCOO(&coo_4);

  // Read the COO from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> readerCOO_4("writer_test_coo_mtx4.mtx");

  auto coo_4_r = readerCOO_4.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_4.get_dimensions(), coo_4_r->get_dimensions());
  EXPECT_EQ(coo_4.get_num_nnz(), coo_4_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo_4.get_row()[i], coo_4_r->get_row()[i]);
    EXPECT_EQ(coo_4.get_col()[i], coo_4_r->get_col()[i]);
    EXPECT_EQ(coo_4.get_vals()[i], coo_4_r->get_vals()[i]);
  }
}

TEST(MTXWriter, WriteCSR) {
  // Initialize a CSR for testing
  int row_10[6]{0, 0, 1, 2, 3, 5};
  int col_10[5]{0, 1, 0, 2, 3};
  float vals_10[5]{0.1, 0.3, 0.2, 0.4, 0.5};

  sparsebase::format::CSR<int, int, float> csr_1(5, 5, row_10, col_10, vals_10,
                                               sparsebase::format::kNotOwned);

  // Write the CSR to a Mtx file with sparsebase
  sparsebase::io::MTXWriter<int, int, float> writerCSR_1("writer_test_csr_mtx.mtx");
  writerCSR_1.WriteCSR(&csr_1);

  // Read the CSR from the Mtx file with sparsebase
  sparsebase::io::MTXReader<int, int, float> readerCSR_1("writer_test_csr_mtx.mtx");

  auto csr_1_r = readerCSR_1.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr_1.get_dimensions(), csr_1_r->get_dimensions());
  EXPECT_EQ(csr_1.get_num_nnz(), csr_1_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr_1.get_col()[i], csr_1_r->get_col()[i]);
    EXPECT_EQ(csr_1.get_vals()[i], csr_1_r->get_vals()[i]);
  }
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr_1.get_row_ptr()[i], csr_1_r->get_row_ptr()[i]);
  }
}
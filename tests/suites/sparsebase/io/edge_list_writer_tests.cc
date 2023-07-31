#include <string>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
#include "sparsebase/io/edge_list_writer.h"
TEST(EdgeListWriter, WriteCOO) {

  // 1 - Weighted & Directed
  // Initialize a weighted and directed COO for testing
  int row_1[5]{1, 5, 2, 3, 4};
  int col_1[5]{5, 1, 6, 7, 8};
  float vals_1[5]{0.1, 0.5,0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo_1(6, 9, 5, row_1, col_1, vals_1,
                                               sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (weighted, directed)
  sparsebase::io::EdgeListWriter<int, int, float> writerCOO_1(
      "writer_test_coo_weighted_directed.edgelist", true);
  writerCOO_1.WriteCOO(&coo_1);

  // Read it using sparsebase EdgeListReader (weighted, directed)
  sparsebase::io::EdgeListReader<int, int, float> readerCOO_1(
      "writer_test_coo_weighted_directed.edgelist",
      true, false, false, false);

  auto coo_1_r = readerCOO_1.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_1.get_dimensions(), coo_1_r->get_dimensions());
  EXPECT_EQ(coo_1.get_num_nnz(), coo_1_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo_1.get_row()[i], coo_1_r->get_row()[i]);
    EXPECT_EQ(coo_1.get_col()[i], coo_1_r->get_col()[i]);
    EXPECT_EQ(coo_1.get_vals()[i], coo_1_r->get_vals()[i]);
  }

  // 2 - Weighted & Undirected
  // Initialize a weighted and undirected COO for testing
  int row_2[4]{1, 2, 3, 4};
  int col_2[4]{5, 6, 7, 8};
  float vals_2[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo_2(5, 9, 4, row_2, col_2, vals_2,
                                                 sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (weighted, undirected)
  sparsebase::io::EdgeListWriter<int, int, float> writerCOO_2(
      "writer_test_coo_weighted_undirected.edgelist", false);
  writerCOO_2.WriteCOO(&coo_2);

  // Read it using sparsebase EdgeListReader (weighted, undirected)
  sparsebase::io::EdgeListReader<int, int, float> readerCOO_2(
      "writer_test_coo_weighted_undirected.edgelist",
      true, false, false, false);

  auto coo_2_r = readerCOO_2.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_2.get_dimensions(), coo_2_r->get_dimensions());
  EXPECT_EQ(coo_2.get_num_nnz(), coo_2_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo_2.get_row()[i], coo_2_r->get_row()[i]);
    EXPECT_EQ(coo_2.get_col()[i], coo_2_r->get_col()[i]);
    EXPECT_EQ(coo_2.get_vals()[i], coo_2_r->get_vals()[i]);
  }

  // 3 - Unweighted & Directed
  // Initialize a unweighted and directed COO for testing
  int row_3[5]{1, 5,2, 3, 4};
  int col_3[5]{5, 1,6, 7, 8};
  sparsebase::format::COO<int, int, void> coo_3(6,9,5, row_3, col_3, nullptr,
                                                 sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (unweighted, directed)
  sparsebase::io::EdgeListWriter<int, int, void> writerCOO_3(
      "writer_test_coo_unweighted_directed.edgelist", true);
  writerCOO_3.WriteCOO(&coo_3);

  // Read it using sparsebase EdgeListReader (unweighted, directed)
  sparsebase::io::EdgeListReader<int, int, void> readerCOO_3(
      "writer_test_coo_unweighted_directed.edgelist",
      false, false, false, false);

  auto coo_3_r = readerCOO_3.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_3.get_dimensions(), coo_3_r->get_dimensions());
  EXPECT_EQ(coo_3.get_num_nnz(), coo_3_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo_3.get_row()[i], coo_3_r->get_row()[i]);
    EXPECT_EQ(coo_3.get_col()[i], coo_3_r->get_col()[i]);
  }

  // 4 - Unweighted & Undirected
  // Initialize a unweighted and undirected COO for testing
  int row_4[4]{1, 2, 3, 4};
  int col_4[4]{5, 6, 7, 8};
  sparsebase::format::COO<int, int, void> coo_4(5,9,4, row_4, col_4, nullptr,
                                                sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (unweighted, undirected)
  sparsebase::io::EdgeListWriter<int, int, void> writerCOO_4(
      "writer_test_coo_unweighted_undirected.edgelist", false);
  writerCOO_4.WriteCOO(&coo_4);

  // Read it using sparsebase EdgeListReader (unweighted, undirected)
  sparsebase::io::EdgeListReader<int, int, void> readerCOO_4(
      "writer_test_coo_unweighted_undirected.edgelist",
      false, false, false, false);

  auto coo_4_r = readerCOO_4.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo_4.get_dimensions(), coo_4_r->get_dimensions());
  EXPECT_EQ(coo_4.get_num_nnz(), coo_4_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo_4.get_row()[i], coo_4_r->get_row()[i]);
    EXPECT_EQ(coo_4.get_col()[i], coo_4_r->get_col()[i]);
  }
}

TEST(EdgeListWriter, CSRWriter) {
  // 1 - Weighted & Directed
  // Initialize a weighted and directed CSR for testing
  int row_ptr_1[5]{0, 2, 3, 3, 4};
  int col_1[4]{0, 2, 1, 3};
  float vals_1[4]{0.1, 0.2, 0.3, 0.4};

  sparsebase::format::CSR<int, int, float> csr_1(4, 4, row_ptr_1, col_1, vals_1,
                                               sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (weighted, directed)
  sparsebase::io::EdgeListWriter<int, int, float> writerCSR_1(
      "writer_test_csr_weighted_directed.edgelist", true);
  writerCSR_1.WriteCSR(&csr_1);

  // Read it using sparsebase EdgeListReader (weighted, directed)
  sparsebase::io::EdgeListReader<int, int, float> readerCSR_1(
      "writer_test_csr_weighted_directed.edgelist",
      true, false, false, false);
  auto csr_1_r = readerCSR_1.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr_1.get_dimensions(), csr_1_r->get_dimensions());
  EXPECT_EQ(csr_1.get_num_nnz(), csr_1_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr_1.get_col()[i], csr_1_r->get_col()[i]);
    EXPECT_EQ(csr_1.get_vals()[i], csr_1_r->get_vals()[i]);
  }

  //Compare the row pointers
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr_1.get_row_ptr()[i], csr_1_r->get_row_ptr()[i]);
  }

  // 2 - Weighted & Undirected
  // Initialize a weighted and undirected CSR for testing
  int row_ptr_2[5]{0, 2, 3, 3, 4};
  int col_2[4]{0, 2, 1, 3};
  float vals_2[4]{0.1, 0.2, 0.3, 0.4};

  sparsebase::format::CSR<int, int, float> csr_2(4, 4, row_ptr_2, col_2, vals_2,
                                                 sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (weighted, directed)
  sparsebase::io::EdgeListWriter<int, int, float> writerCSR_2(
      "writer_test_csr_weighted_undirected.edgelist", false);
  writerCSR_2.WriteCSR(&csr_2);

  // Read it using sparsebase EdgeListReader (weighted, undirected)
  sparsebase::io::EdgeListReader<int, int, float> readerCSR_2(
      "writer_test_csr_weighted_undirected.edgelist",
      true, true, false, false);
  auto csr_2_r = readerCSR_2.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr_2.get_dimensions(), csr_2_r->get_dimensions());
  EXPECT_EQ(csr_2.get_num_nnz(), csr_2_r->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr_2.get_col()[i], csr_2_r->get_col()[i]);
    EXPECT_EQ(csr_2.get_vals()[i], csr_2_r->get_vals()[i]);
  }

  //Compare the row pointers
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr_2.get_row_ptr()[i], csr_2_r->get_row_ptr()[i]);
  }

  // 3 - Unweighted & Directed
  // Initialize a unweighted and directed CSR for testing
  int row_ptr_3[5]{0, 2, 3, 3, 4};
  int col_3[4]{0, 2, 1, 3};

  sparsebase::format::CSR<int, int, void> csr_3(4, 4, row_ptr_3, col_3, nullptr,
                                                 sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (unweighted, directed)
  sparsebase::io::EdgeListWriter<int, int, void> writerCSR_3(
      "writer_test_csr_unweighted_directed.edgelist", true);
  writerCSR_3.WriteCSR(&csr_3);

  // Read it using sparsebase EdgeListReader (unweighted, directed)
  sparsebase::io::EdgeListReader<int, int, void> readerCSR_3(
      "writer_test_csr_unweighted_directed.edgelist",
      false, false, false, false);
  auto csr_3_r = readerCSR_3.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr_3.get_dimensions(), csr_3_r->get_dimensions());
  EXPECT_EQ(csr_3.get_num_nnz(), csr_3_r->get_num_nnz());

  // Compare the col arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr_3.get_col()[i], csr_3_r->get_col()[i]);
  }

  //Compare the row pointers
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr_3.get_row_ptr()[i], csr_3_r->get_row_ptr()[i]);
  }

  // 4 - Unweighted & Undirected
  // Initialize a unweighted and undirected CSR for testing
  int row_ptr_4[5]{0, 2, 3, 3, 4};
  int col_4[4]{0, 2, 1, 3};

  sparsebase::format::CSR<int, int, void> csr_4(4, 4, row_ptr_4, col_4, nullptr,
                                                sparsebase::format::kNotOwned);

  // Write it using sparsebase EdgeListWriter (unweighted, undirected)
  sparsebase::io::EdgeListWriter<int, int, void> writerCSR_4(
      "writer_test_csr_unweighted_undirected.edgelist", false);
  writerCSR_4.WriteCSR(&csr_4);

  // Read it using sparsebase EdgeListReader (unweighted, undirected)
  sparsebase::io::EdgeListReader<int, int, void> readerCSR_4(
      "writer_test_csr_unweighted_undirected.edgelist",
      false, true, false, false);
  auto csr_4_r = readerCSR_4.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr_4.get_dimensions(), csr_4_r->get_dimensions());
  EXPECT_EQ(csr_4.get_num_nnz(), csr_4_r->get_num_nnz());

  // Compare the col arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr_4.get_col()[i], csr_4_r->get_col()[i]);
  }

  //Compare the row pointers
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr_4.get_row_ptr()[i], csr_4_r->get_row_ptr()[i]);
  }
}

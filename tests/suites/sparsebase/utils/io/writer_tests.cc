#include "sparsebase/sparsebase.h"
#include "gtest/gtest.h"
#include <string>

TEST(IOBase, WriteArrayToBinary) {

  // Initialize an array for testing
  int array[5]{1, 2, 3, 4, 5};
  sparsebase::format::Array<int> sbArray(5, array,
                                         sparsebase::format::kNotOwned);

  // Write the array to a binary file using sparsebase
  utils::io::IOBase::WriteArrayToBinary(&sbArray, "test_order_one.bin");

  // Read the array from the binary file using sparsebase
  sparsebase::utils::io::BinaryReaderOrderOne<int> readerOrderOne(
      "test_order_one.bin");
  auto array2 = readerOrderOne.ReadArray();

  // Compare the arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(array[i], array2->get_vals()[i]);
  }
}

TEST(IOBase, WriteCOOToBinary) {

  // Initialize a COO for testing
  int row[4]{1, 2, 3, 4};
  int col[4]{5, 6, 7, 8};
  float vals[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  utils::io::IOBase::WriteCOOToBinary(
      &coo, "writer_test_order_two_coo.bin");

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
      "writer_test_order_two_coo.bin");
  auto coo2 = readerOrderTwo.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo.get_dimensions(), coo2->get_dimensions());
  EXPECT_EQ(coo.get_num_nnz(), coo2->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo.get_row()[i], coo2->get_row()[i]);
    EXPECT_EQ(coo.get_col()[i], coo2->get_col()[i]);
    EXPECT_EQ(coo.get_vals()[i], coo2->get_vals()[i]);
  }
}

TEST(IOBase, WriteBinaryToCSR) {

  // Initialize a CSR for testing
  int row_ptr[5]{0, 2, 3, 3, 4};
  int col[4]{0, 2, 1, 3};
  float vals[4]{0.1, 0.2, 0.3, 0.4};

  sparsebase::format::CSR<int, int, float> csr(4, 4, row_ptr, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  utils::io::IOBase::WriteCSRToBinary(&csr,
      "writer_test_order_two_csr.bin");

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
      "writer_test_order_two_csr.bin");
  auto csr2 = readerOrderTwo.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr.get_dimensions(), csr2->get_dimensions());
  EXPECT_EQ(csr.get_num_nnz(), csr2->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr.get_col()[i], csr2->get_col()[i]);
    EXPECT_EQ(csr.get_vals()[i], csr2->get_vals()[i]);
  }

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr.get_row_ptr()[i], csr2->get_row_ptr()[i]);
  }
}
TEST(BinaryOrderOneWriter, Basics) {

  // Initialize an array for testing
  int array[5]{1, 2, 3, 4, 5};
  sparsebase::format::Array<int> sbArray(5, array,
                                         sparsebase::format::kNotOwned);

  // Write the array to a binary file using sparsebase
  sparsebase::utils::io::BinaryWriterOrderOne<int> writerOrderOne(
      "test_order_one.bin");
  writerOrderOne.WriteArray(&sbArray);

  // Read the array from the binary file using sparsebase
  sparsebase::utils::io::BinaryReaderOrderOne<int> readerOrderOne(
      "test_order_one.bin");
  auto array2 = readerOrderOne.ReadArray();

  // Compare the arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(array[i], array2->get_vals()[i]);
  }
}

TEST(BinaryOrderTwoWriter, COO) {

  // Initialize a COO for testing
  int row[4]{1, 2, 3, 4};
  int col[4]{5, 6, 7, 8};
  float vals[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "writer_test_order_two_coo.bin");
  writerOrderTwo.WriteCOO(&coo);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
      "writer_test_order_two_coo.bin");
  auto coo2 = readerOrderTwo.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo.get_dimensions(), coo2->get_dimensions());
  EXPECT_EQ(coo.get_num_nnz(), coo2->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(coo.get_row()[i], coo2->get_row()[i]);
    EXPECT_EQ(coo.get_col()[i], coo2->get_col()[i]);
    EXPECT_EQ(coo.get_vals()[i], coo2->get_vals()[i]);
  }
}

TEST(BinaryOrderTwoWriter, CSR) {

  // Initialize a CSR for testing
  int row_ptr[5]{0, 2, 3, 3, 4};
  int col[4]{0, 2, 1, 3};
  float vals[4]{0.1, 0.2, 0.3, 0.4};

  sparsebase::format::CSR<int, int, float> csr(4, 4, row_ptr, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "writer_test_order_two_csr.bin");
  writerOrderTwo.WriteCSR(&csr);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
      "writer_test_order_two_csr.bin");
  auto csr2 = readerOrderTwo.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr.get_dimensions(), csr2->get_dimensions());
  EXPECT_EQ(csr.get_num_nnz(), csr2->get_num_nnz());

  // Compare the underlying arrays
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(csr.get_col()[i], csr2->get_col()[i]);
    EXPECT_EQ(csr.get_vals()[i], csr2->get_vals()[i]);
  }

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr.get_row_ptr()[i], csr2->get_row_ptr()[i]);
  }
}
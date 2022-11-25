#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "../io/reader_data.inc"
#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"

TEST(IOBase, ReadBinaryToCOO) {
  // Initialize a COO for testing
  int row[4]{1, 2, 3, 4};
  int col[4]{5, 6, 7, 8};
  float vals[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "reader_test_order_two_coo.bin");
  writerOrderTwo.WriteCOO(&coo);

  // Read the COO from the binary file with sparsebase
  auto coo2 = sparsebase::io::IOBase::ReadBinaryToCOO<int, int, float>(
      "reader_test_order_two_coo.bin");

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

TEST(IOBase, ReadBinaryToCSR) {
  // Initialize a CSR for testing
  int row_ptr[5]{0, 2, 3, 3, 4};
  int col[4]{0, 2, 1, 3};
  float vals[4]{0.1, 0.2, 0.3, 0.4};

  sparsebase::format::CSR<int, int, float> csr(4, 4, row_ptr, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "reader_test_order_two_csr.bin");
  writerOrderTwo.WriteCSR(&csr);

  // Read the COO from the binary file with sparsebase
  auto csr2 = sparsebase::io::IOBase::ReadBinaryToCSR<int, int, float>(
      "reader_test_order_two_csr.bin");

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
TEST(IOBase, ReadBinaryToArray) {
  // Initialize an array for testing
  int array[5]{1, 2, 3, 4, 5};
  sparsebase::format::Array<int> sbArray(5, array,
                                         sparsebase::format::kNotOwned);

  // Write the array to a binary file using sparsebase
  sparsebase::io::BinaryWriterOrderOne<int> writerOrderOne(
      "test_order_one.bin");
  writerOrderOne.WriteArray(&sbArray);

  // Read the array from the binary file using sparsebase
  auto array2 =
      sparsebase::io::IOBase::ReadBinaryToArray<int>("test_order_one.bin");

  // Compare the arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(array[i], array2->get_vals()[i]);
  }
}

TEST(IOBase, ReadPigoMTXToCSR) {
  // Write the mtx data to a file
  std::ofstream ofs("test_pigo.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values_pigo.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  auto csr = sparsebase::io::IOBase::ReadPigoMTXToCSR<int, int, int>(
      "test_pigo.mtx", false);

  // Check the dimensions
  EXPECT_EQ(csr->get_dimensions()[0], 5);
  EXPECT_EQ(csr->get_dimensions()[1], 5);
  EXPECT_EQ(csr->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(csr->get_row_ptr(), nullptr);
  EXPECT_NE(csr->get_col(), nullptr);

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr->get_col()[i], col[i]);
  }

  auto csr2 = sparsebase::io::IOBase::ReadPigoMTXToCSR<int, int, float>(
      "test_values_pigo.mtx", true);

  // Check the dimensions
  EXPECT_EQ(csr2->get_dimensions()[0], 5);
  EXPECT_EQ(csr2->get_dimensions()[1], 5);
  EXPECT_EQ(csr2->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(csr2->get_vals(), nullptr);
  EXPECT_NE(csr2->get_row_ptr(), nullptr);
  EXPECT_NE(csr2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr2->get_row_ptr()[i], row_ptr[i]);
  }

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr2->get_col()[i], col[i]);
    EXPECT_EQ(csr2->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadPigoMTXReadToCOO) {
  // Write the mtx data to a file
  std::ofstream ofs("test_pigo.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values_pigo.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  auto coo = sparsebase::io::IOBase::ReadPigoMTXToCOO<int, int, int>(
      "test_pigo.mtx", false);

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  auto coo2 = sparsebase::io::IOBase::ReadPigoMTXToCOO<int, int, float>(
      "test_values_pigo.mtx", true);

  // Check the dimensions
  EXPECT_EQ(coo2->get_dimensions()[0], 5);
  EXPECT_EQ(coo2->get_dimensions()[1], 5);
  EXPECT_EQ(coo2->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo2->get_vals(), nullptr);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadMTXReadToCSR) {
  // Write the mtx data to a file
  std::ofstream ofs("test.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  auto csr = sparsebase::io::IOBase::ReadMTXToCSR<int, int, int>(
      "test_pigo.mtx", true);

  // Check the dimensions
  EXPECT_EQ(csr->get_dimensions()[0], 5);
  EXPECT_EQ(csr->get_dimensions()[1], 5);
  EXPECT_EQ(csr->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(csr->get_row_ptr(), nullptr);
  EXPECT_NE(csr->get_col(), nullptr);

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr->get_col()[i], col[i]);
  }

  auto csr2 = sparsebase::io::IOBase::ReadMTXToCSR<int, int, float>(
      "test_values_pigo.mtx", true);

  // Check the dimensions
  EXPECT_EQ(csr2->get_dimensions()[0], 5);
  EXPECT_EQ(csr2->get_dimensions()[1], 5);
  EXPECT_EQ(csr2->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(csr2->get_vals(), nullptr);
  EXPECT_NE(csr2->get_row_ptr(), nullptr);
  EXPECT_NE(csr2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr2->get_row_ptr()[i], row_ptr[i]);
  }

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr2->get_col()[i], col[i]);
    EXPECT_EQ(csr2->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadMTXReadToCOO) {
  // Write the mtx data to a file
  std::ofstream ofs("test.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  // Read non-weighted file using sparsebase
  auto coo =
      sparsebase::io::IOBase::ReadMTXToCOO<int, int, int>("test.mtx", true);

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  // Read the weighted file using sparsebase
  auto coo2 = sparsebase::io::IOBase::ReadMTXToCOO<int, int, float>(
      "test_values.mtx", true);

  // Check the dimensions
  EXPECT_EQ(coo2->get_dimensions()[0], 5);
  EXPECT_EQ(coo2->get_dimensions()[1], 5);
  EXPECT_EQ(coo2->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo2->get_vals(), nullptr);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadEdgeListToCSR) {
  // Write the edge list data to a file
  std::ofstream ofs("test.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  // Read it using sparsebase (undirected)
  auto csr =
      sparsebase::io::IOBase::ReadEdgeListToCSR<int, int, int>("test.edges");

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(csr->get_num_nnz(), 10);
  EXPECT_NE(csr->get_row_ptr(), nullptr);
  EXPECT_NE(csr->get_col(), nullptr);

  // To test integrity we create an edge_set from the known values
  // And check if every edge read is a member of that
  // This is done due to the fact that this is an undirected read
  std::set<std::pair<int, int>> edge_set;
  for (int i = 0; i < 5; i++) {
    edge_set.emplace(row[i], col[i]);
    edge_set.emplace(col[i], row[i]);
  }
  for (int i = 0; i < 5; i++) {
    for (int j = csr->get_row_ptr()[i]; j < csr->get_row_ptr()[i + 1]; j++) {
      std::pair<int, int> p(i, csr->get_col()[j]);
      EXPECT_NE(edge_set.find(p), edge_set.end());
    }
  }

  // Read it using sparsebase (directed)
  auto csr2 = sparsebase::io::IOBase::ReadEdgeListToCSR<int, int, float>(
      "test.edges", false, false, false);

  // Check the dimensions
  EXPECT_EQ(csr2->get_num_nnz(), 5);
  EXPECT_NE(csr2->get_row_ptr(), nullptr);
  EXPECT_NE(csr2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr2->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr2->get_col()[i], col[i]);
  }

  // Read it using sparsebase (weighted, directed)
  auto csr3 = sparsebase::io::IOBase::ReadEdgeListToCSR<int, int, float>(
      "test_values.edges", true, false, false);

  // Check the dimensions
  EXPECT_EQ(csr3->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(csr3->get_vals(), nullptr);
  EXPECT_NE(csr3->get_row_ptr(), nullptr);
  EXPECT_NE(csr3->get_col(), nullptr);

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr3->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr3->get_col()[i], col[i]);
    EXPECT_EQ(csr3->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadEdgeListToCOO) {
  // Write the edge list data to a file
  std::ofstream ofs("test.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  // Read it using sparsebase (undirected)
  auto coo =
      sparsebase::io::IOBase::ReadEdgeListToCOO<int, int, int>("test.edges");

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo->get_num_nnz(), 10);
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // To test integrity we create an edge_set from the known values
  // And check if every edge read is a member of that
  // This is done due to the fact that this is an undirected read
  std::set<std::pair<int, int>> edge_set;
  for (int i = 0; i < 5; i++) {
    edge_set.emplace(row[i], col[i]);
    edge_set.emplace(col[i], row[i]);
  }
  for (int i = 0; i < 10; i++) {
    std::pair<int, int> p(coo->get_row()[i], coo->get_col()[i]);
    EXPECT_NE(edge_set.find(p), edge_set.end());
  }

  // Read it using sparsebase (directed)
  auto coo2 = sparsebase::io::IOBase::ReadEdgeListToCOO<int, int, int>(
      "test.edges", false, false, false);

  // Check the dimensions
  EXPECT_EQ(coo2->get_num_nnz(), 5);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
  }

  // Read it using sparsebase (weighted, directed)
  auto coo3 = sparsebase::io::IOBase::ReadEdgeListToCOO<int, int, float>(
      "test_values.edges", true, false, false);

  // Check the dimensions
  EXPECT_EQ(coo3->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo3->get_vals(), nullptr);
  EXPECT_NE(coo3->get_row(), nullptr);
  EXPECT_NE(coo3->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo3->get_row()[i], row[i]);
    EXPECT_EQ(coo3->get_col()[i], col[i]);
    EXPECT_EQ(coo3->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadPigoEdgeListToCOO) {
  // Write the edge list data to a file
  std::ofstream ofs("test_pigo.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values_pigo.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  auto coo = sparsebase::io::IOBase::ReadPigoEdgeListToCOO<int, int, int>(
      "test_pigo.edges", false);

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo->get_num_nnz(), 5);
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  auto coo2 = sparsebase::io::IOBase::ReadPigoEdgeListToCOO<int, int, float>(
      "test_values_pigo.edges", true);

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo2->get_num_nnz(), 5);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);
  EXPECT_NE(coo2->get_vals(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals[i]);
  }
}

TEST(IOBase, ReadPigoEdgeListToCSR) {
  // Write the edge list data to a file
  std::ofstream ofs("test_pigo.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values_pigo.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  auto csr = sparsebase::io::IOBase::ReadPigoEdgeListToCSR<int, int, int>(
      "test_pigo.edges", false);

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(csr->get_num_nnz(), 5);
  EXPECT_NE(csr->get_row_ptr(), nullptr);
  EXPECT_NE(csr->get_col(), nullptr);

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr->get_col()[i], col[i]);
  }

  auto csr2 = sparsebase::io::IOBase::ReadPigoEdgeListToCSR<int, int, float>(
      "test_values_pigo.edges", true);

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(csr2->get_num_nnz(), 5);
  EXPECT_NE(csr2->get_row_ptr(), nullptr);
  EXPECT_NE(csr2->get_col(), nullptr);
  EXPECT_NE(csr2->get_vals(), nullptr);

  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(csr2->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr2->get_col()[i], col[i]);
    EXPECT_EQ(csr2->get_vals()[i], vals[i]);
  }
}
TEST(IOBase, ReadToOrderOne) {
  std::ofstream ofs("one_col.mtx");
  ofs << mtx_data_one_col_with_values;
  ofs.close();
  // Read one column file
  auto array =
      sparsebase::io::IOBase::ReadMTXToArray<int, int, float>("one_col.mtx");
  // Check that the arrays are populated
  EXPECT_NE(array->get_vals(), nullptr);
  for (int i = 0; i < 10; i++) std::cout << array->get_vals()[i] << " ";
  std::cout << std::endl;

  // Check the integrity and order of data
  for (int i = 0; i < one_row_one_col_length; i++) {
    EXPECT_EQ(array->get_vals()[i], one_row_one_col_vals[i]);
  }
}
TEST(IOBase, WriteArrayToBinary) {
  // Initialize an array for testing
  int array[5]{1, 2, 3, 4, 5};
  sparsebase::format::Array<int> sbArray(5, array,
                                         sparsebase::format::kNotOwned);

  // Write the array to a binary file using sparsebase
  sparsebase::io::IOBase::WriteArrayToBinary(&sbArray, "test_order_one.bin");

  // Read the array from the binary file using sparsebase
  sparsebase::io::BinaryReaderOrderOne<int> readerOrderOne(
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
  sparsebase::io::IOBase::WriteCOOToBinary(&coo,
                                           "writer_test_order_two_coo.bin");

  // Read the COO from the binary file with sparsebase
  sparsebase::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
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
  sparsebase::io::IOBase::WriteCSRToBinary(&csr,
                                           "writer_test_order_two_csr.bin");

  // Read the COO from the binary file with sparsebase
  sparsebase::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
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

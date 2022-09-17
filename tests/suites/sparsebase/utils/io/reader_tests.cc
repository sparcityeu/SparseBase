#include "sparsebase/sparsebase.h"
#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

const std::string mtx_data =
    R"(%%MatrixMarket matrix coordinate pattern general
%This is a comment
5 5 5
2 1
4 1
3 2
5 3
5 4
)";

const std::string mtx_symm_data =
    R"(%%MatrixMarket matrix coordinate pattern symmetric
%This is a comment
5 5 6
1 1
2 1
4 1
3 2
5 3
5 4
)";

const std::string mtx_data_with_values =
    R"(%%MatrixMarket matrix coordinate real general
%This is a comment
5 5 5
2 1 0.1
4 1 0.2
3 2 0.3
5 3 0.4
5 4 0.5
)";

const std::string mtx_data_with_values_array =
    R"(%%MatrixMarket matrix array real general
%This is a comment
5 5
0
0
0
0
0
0.1
0
0
0
0
0
0.3
0
0
0
0.2
0
0
0
0
0
0
0.4
0.5
0
)";

const std::string mtx_data_array =
    R"(%%MatrixMarket matrix array real general
%This is a comment
5 5
0
0
0
0
0
0.1
0
0
0
0
0
0.3
0
0
0
0.2
0
0
0
0
0
0
0.4
0.5
0
)";


const std::string mtx_symm_data_with_values =
    R"(%%MatrixMarket matrix coordinate real symmetric
%This is a comment
5 5 6
1 1 0.7
2 1 0.1
4 1 0.2
3 2 0.3
5 3 0.4
5 4 0.5
)";

const std::string mtx_data_one_col_with_values =
    R"(%%MatrixMarket matrix coordinate real general
%This is a comment
10 1 5
2 1 0.1
4 1 0.3
5 1 0.2
6 1 0.4
9 1 0.5
)";

const std::string mtx_data_one_row_with_values =
    R"(%%MatrixMarket matrix coordinate real general
%This is a comment
1 10 5
1 2 0.1
1 4 0.3
1 5 0.2
1 6 0.4
1 9 0.5
)";

const std::string mtx_array_data_one_col_with_values =
    R"(%%MatrixMarket matrix array real general
%This is a comment
10 1
0
0.1
0
0.3
0.2
0.4
0
0
0.5
0
)";

const std::string mtx_array_data_one_row_with_values =
    R"(%%MatrixMarket matrix array real general
%This is a comment
1 10
0
0.1
0
0.3
0.2
0.4
0
0
0.5
0
)";

const std::string edge_list_data = R"(1 0
3 0
2 1
4 2
4 3
)";

const std::string edge_list_data_with_values = R"(1 0 0.1
3 0 0.2
2 1 0.3
4 2 0.4
4 3 0.5
)";

// Both files should result in the COO arrays below
// (With the convert_to_zero_index option set to true for mtx)
int row_ptr[6]{0, 0, 1, 2, 3,  5};
int row[5]{1, 2, 3, 4, 4};
int col[5]{0, 1, 0, 2, 3};
int one_row_one_col[5]{1, 3, 4, 5, 8};
float one_row_one_col_vals[10]{0, 0.1, 0, 0.3, 0.2, 0.4, 0, 0, 0.5, 0};
int one_row_one_col_length = 10;
float vals[5]{0.1, 0.3, 0.2, 0.4, 0.5};
int row_ptr_symm[6]{0,             3,        5,        7,        9,       11};
int row_symm[11]   {0,   0,   0,   1,   1,   2,   2,   3,   3,   4,   4};
int col_symm[11]   {0,   1,   3,   0,   2,   1,   4,   0,   4,   2,   3};
float vals_symm[11]{0.7, 0.1, 0.2, 0.1, 0.3, 0.3, 0.4, 0.2, 0.5, 0.4, 0.5};

void checkArrayReading(std::string filename) {

  // Read one column file
  sparsebase::utils::io::MTXReader<int, int, float> reader(filename, true);
  auto array = reader.ReadArray();

  // Check the dimensions
  EXPECT_EQ(array->get_dimensions()[0], one_row_one_col_length);
  EXPECT_EQ(array->get_num_nnz(), one_row_one_col_length);
}

TEST(IOBase, ReadBinaryToCOO) {

  // Initialize a COO for testing
  int row[4]{1, 2, 3, 4};
  int col[4]{5, 6, 7, 8};
  float vals[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "reader_test_order_two_coo.bin");
  writerOrderTwo.WriteCOO(&coo);

  // Read the COO from the binary file with sparsebase
  auto coo2 = utils::io::IOBase::ReadBinaryToCOO<int, int, float>("reader_test_order_two_coo.bin");

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
  sparsebase::utils::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "reader_test_order_two_csr.bin");
  writerOrderTwo.WriteCSR(&csr);

  // Read the COO from the binary file with sparsebase
  auto csr2 = utils::io::IOBase::ReadBinaryToCSR<int, int, float>("reader_test_order_two_csr.bin");

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
  sparsebase::utils::io::BinaryWriterOrderOne<int> writerOrderOne(
      "test_order_one.bin");
  writerOrderOne.WriteArray(&sbArray);

  // Read the array from the binary file using sparsebase
  auto array2 = utils::io::IOBase::ReadBinaryToArray<int>(
      "test_order_one.bin");

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

  auto csr = utils::io::IOBase::ReadPigoMTXToCSR<int, int, int>("test_pigo.mtx",
                                                                false);

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

  auto csr2 = utils::io::IOBase::ReadPigoMTXToCSR<int, int, float>("test_values_pigo.mtx",
                                                                   true);

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

  auto coo = utils::io::IOBase::ReadPigoMTXToCOO<int, int, int>("test_pigo.mtx",
                                                                false);

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

  auto coo2 = utils::io::IOBase::ReadPigoMTXToCOO<int, int, float>("test_values_pigo.mtx",
                                                                true);

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

  auto csr = utils::io::IOBase::ReadMTXToCSR<int, int, int>("test_pigo.mtx",
                                                                true);

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

  auto csr2 = utils::io::IOBase::ReadMTXToCSR<int, int, float>("test_values_pigo.mtx",
                                                                   true);

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
  auto coo = sparsebase::utils::io::IOBase::ReadMTXToCOO<int, int, int>("test.mtx", true);

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
  auto coo2 = sparsebase::utils::io::IOBase::ReadMTXToCOO<int, int, float>("test_values.mtx", true);

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
  auto  csr = utils::io::IOBase::ReadEdgeListToCSR<int, int, int>("test.edges");

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
    for (int j =csr->get_row_ptr()[i]; j < csr->get_row_ptr()[i+1]; j++) {
      std::pair<int, int> p(i, csr->get_col()[j]);
      EXPECT_NE(edge_set.find(p), edge_set.end());
    }
  }

  // Read it using sparsebase (directed)
  auto csr2 = utils::io::IOBase::ReadEdgeListToCSR<int, int, float>("test.edges", false, false, false);

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
  auto csr3 = utils::io::IOBase::ReadEdgeListToCSR<int, int, float>("test_values.edges", true, false, false);

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
  auto coo = utils::io::IOBase::ReadEdgeListToCOO<int, int, int>("test.edges");

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
  auto coo2 = utils::io::IOBase::ReadEdgeListToCOO<int, int, int>("test.edges", false, false, false);

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
  auto coo3 = utils::io::IOBase::ReadEdgeListToCOO<int, int, float>("test_values.edges", true, false, false);

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

  auto coo = utils::io::IOBase::ReadPigoEdgeListToCOO<int, int, int>(
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

  auto coo2 = utils::io::IOBase::ReadPigoEdgeListToCOO<int, int, float>(
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

  auto csr = utils::io::IOBase::ReadPigoEdgeListToCSR<int, int, int>(
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

  auto csr2 = utils::io::IOBase::ReadPigoEdgeListToCSR<int, int, float>(
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
  auto array = utils::io::IOBase::ReadMTXToArray<int, int, float>("one_col.mtx");
  // Check that the arrays are populated
  EXPECT_NE(array->get_vals(), nullptr);
  for (int i= 0; i< 10; i++) std::cout << array->get_vals()[i] <<" ";
  std::cout << std::endl;

  // Check the integrity and order of data
  for (int i =0; i< one_row_one_col_length; i++){
    EXPECT_EQ(array->get_vals()[i], one_row_one_col_vals[i]);
  }
}
TEST(MTXReader, ArrayOneCol) {

  std::ofstream ofs("one_col.mtx");
  ofs << mtx_data_one_col_with_values;
  ofs.close();
  // Write the mtx data to a file
  checkArrayReading("one_col.mtx");
}

TEST(MTXReader, ArrayOneRow) {
  std::ofstream ofs("one_row.mtx");
  ofs << mtx_data_one_row_with_values;
  ofs.close();
  // Write the mtx data to a file
  checkArrayReading("one_row.mtx");

}
TEST(MTXReader, ArrayOneRowArray) {
  std::ofstream ofs("one_row_array.mtx");
  ofs << mtx_array_data_one_row_with_values;
  ofs.close();
  // Write the mtx data to a file
  checkArrayReading("one_row_array.mtx");
}

TEST(MTXReader, ArrayOneColArray) {
  std::ofstream ofs("one_col_array.mtx");
  ofs << mtx_array_data_one_col_with_values;
  ofs.close();
  // Write the mtx data to a file
  checkArrayReading("one_col_array.mtx");
}

// TODO Add this back once void as a value type is legal again
//TEST(MTXReader, ReadingWeightedIntoVoidValues) {
//
//  // Write the mtx data with values to a file
//  std::ofstream ofs2("test_values.mtx");
//  ofs2 << mtx_data_with_values;
//  ofs2.close();
//
//  // Read the weighted file using sparsebase
//  EXPECT_THROW((sparsebase::utils::io::MTXReader<int, int, void>(
//                   "test_values.mtx", true)),
//               sparsebase::utils::ReaderException);
//}
TEST(MTXReader, BasicsGeneral) {

  // Write the mtx data to a file
  std::ofstream ofs("test.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  // Read non-weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int, int, int> reader("test.mtx");
  auto coo = reader.ReadCOO();

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
  sparsebase::utils::io::MTXReader<int, int, float> reader2("test_values.mtx",
                                                            true);
  auto coo2 = reader2.ReadCOO();

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

TEST(MTXReader, BasicsArray) {

  // Write the mtx data to a file
  std::ofstream ofs("test_arr.mtx");
  ofs << mtx_data_array;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_arr_values.mtx");
  ofs2 << mtx_data_with_values_array;
  ofs2.close();

  // Read non-weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int, int, int> reader("test.mtx");
  auto coo = reader.ReadCOO();

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
  sparsebase::utils::io::MTXReader<int, int, float> reader2("test_values.mtx",
                                                            true);
  auto coo2 = reader2.ReadCOO();

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


TEST(MTXReader, BasicsSymmetric) {

  // Write the mtx data to a file
  std::ofstream ofs("test_symm.mtx");
  ofs << mtx_symm_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_symm_values.mtx");
  ofs2 << mtx_symm_data_with_values;
  ofs2.close();

  // Read non-weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int, int, int> reader("test_symm.mtx");
  auto coo = reader.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 11);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 11; i++) {
    EXPECT_EQ(coo->get_row()[i], row_symm[i]);
    EXPECT_EQ(coo->get_col()[i], col_symm[i]);
  }

  // Read the weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int, int, float> reader2("test_symm_values.mtx",
                                                            true);
  auto coo2 = reader2.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo2->get_dimensions()[0], 5);
  EXPECT_EQ(coo2->get_dimensions()[1], 5);
  EXPECT_EQ(coo2->get_num_nnz(), 11);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo2->get_vals(), nullptr);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 11; i++) {
    EXPECT_EQ(coo2->get_row()[i], row_symm[i]);
    EXPECT_EQ(coo2->get_col()[i], col_symm[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals_symm[i]);
  }
}

TEST(EdgeListReader, Basics) {

  // Write the edge list data to a file
  std::ofstream ofs("test.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  // Read it using sparsebase (undirected)
  sparsebase::utils::io::EdgeListReader<int, int, int> reader1("test.edges");
  auto coo = reader1.ReadCOO();

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
  sparsebase::utils::io::EdgeListReader<int, int, int> reader2(
      "test.edges", false, false, false, false);
  auto coo2 = reader2.ReadCOO();

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
  sparsebase::utils::io::EdgeListReader<int, int, float> reader3(
      "test_values.edges", true, false, false, false);
  auto coo3 = reader3.ReadCOO();

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
TEST(PigoMTXReader, CSR) {
  // Write the mtx data to a file
  std::ofstream ofs("test_pigo.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values_pigo.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  sparsebase::utils::io::PigoMTXReader<int, int, int> reader("test_pigo.mtx",
                                                             false);
  auto csr = reader.ReadCSR();

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

  sparsebase::utils::io::PigoMTXReader<int, int, float> reader2(
      "test_values_pigo.mtx", true);
  auto csr2 = reader2.ReadCSR();

  // Check the dimensions
  EXPECT_EQ(csr2->get_dimensions()[0], 5);
  EXPECT_EQ(csr2->get_dimensions()[1], 5);
  EXPECT_EQ(csr2->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(csr2->get_vals(), nullptr);
  EXPECT_NE(csr2->get_row_ptr(), nullptr);
  EXPECT_NE(csr2->get_col(), nullptr);


  for (int i = 0; i < 5; i++) {
  EXPECT_EQ(csr2->get_row_ptr()[i], row_ptr[i]);
  }
  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(csr2->get_col()[i], col[i]);
    EXPECT_EQ(csr2->get_vals()[i], vals[i]);
  }
}

TEST(PigoMTXReader, Basics) {
  // Write the mtx data to a file
  std::ofstream ofs("test_pigo.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values_pigo.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  sparsebase::utils::io::PigoMTXReader<int, int, int> reader("test_pigo.mtx",
                                                             false);
  auto coo = reader.ReadCOO();

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

  sparsebase::utils::io::PigoMTXReader<int, int, float> reader2(
      "test_values_pigo.mtx", true);
  auto coo2 = reader2.ReadCOO();

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

TEST(PigoEdgeListReader, Basics) {
  // Write the edge list data to a file
  std::ofstream ofs("test_pigo.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values_pigo.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  sparsebase::utils::io::PigoEdgeListReader<int, int, int> reader(
      "test_pigo.edges", false);
  auto coo = reader.ReadCOO();

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo->get_num_nnz(), 5);
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  sparsebase::utils::io::PigoEdgeListReader<int, int, float> reader2(
      "test_values_pigo.edges", true);
  auto coo2 = reader2.ReadCOO();

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

TEST(BinaryOrderOneReader, Basics) {

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

TEST(BinaryOrderTwoReader, COO) {

  // Initialize a COO for testing
  int row[4]{1, 2, 3, 4};
  int col[4]{5, 6, 7, 8};
  float vals[4]{0.1, 0.2, 0.3, 0.4};
  sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "reader_test_order_two_coo.bin");
  writerOrderTwo.WriteCOO(&coo);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
      "reader_test_order_two_coo.bin");
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

TEST(BinaryOrderTwoReader, CSR) {
  // Initialize a CSR for testing
  int row_ptr[5]{0, 2, 3, 3, 4};
  int col[4]{0, 2, 1, 3};
  float vals[4]{0.1, 0.2, 0.3, 0.4};

  sparsebase::format::CSR<int, int, float> csr(4, 4, row_ptr, col, vals,
                                               sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
      "reader_test_order_two_csr.bin");
  writerOrderTwo.WriteCSR(&csr);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int, int, float> readerOrderTwo(
      "reader_test_order_two_csr.bin");
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
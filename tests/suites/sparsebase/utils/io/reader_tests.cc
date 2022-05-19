#include "gtest/gtest.h"
#include <string>
#include <fstream>
#include "sparsebase/sparsebase.h"
#include <iostream>
#include <unordered_set>

const std::string mtx_data = R"(%%MatrixMarket matrix coordinate pattern symmetric
%This is a comment
5 5 5
2 1
4 1
3 2
5 3
5 4
)";



const std::string mtx_data_with_values = R"(%%MatrixMarket matrix coordinate pattern symmetric
%This is a comment
5 5 5
2 1 0.1
4 1 0.2
3 2 0.3
5 3 0.4
5 4 0.5
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
int row[5]{1,2,3,4,4};
int col[5]{0,1,0,2,3};
float vals[5]{0.1, 0.3, 0.2, 0.4, 0.5};


TEST(MTXReader, Basics){

  // Write the mtx data to a file
  std::ofstream ofs("test.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  // Read non-weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int,int,int> reader("test.mtx");
  auto coo = reader.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  // Read the weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int,int,float> reader2("test_values.mtx", true);
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
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals[i]);
  }
}

TEST(EdgeListReader, Basics){

  // Write the edge list data to a file
  std::ofstream ofs("test.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  // Read it using sparsebase (undirected)
  sparsebase::utils::io::EdgeListReader<int,int,int> reader1("test.edges");
  auto coo = reader1.ReadCOO();

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo->get_num_nnz(), 10);
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // To test integrity we create an edge_set from the known values
  // And check if every edge read is a member of that
  // This is done due to the fact that this is an undirected read
  std::set<std::pair<int,int>> edge_set;
  for(int i=0; i<5; i++){
    edge_set.emplace(row[i], col[i]);
    edge_set.emplace(col[i], row[i]);
  }
  for(int i=0; i<10; i++){
    std::pair<int,int> p(coo->get_row()[i], coo->get_col()[i]);
    EXPECT_NE(edge_set.find(p), edge_set.end());
  }

  // Read it using sparsebase (directed)
  sparsebase::utils::io::EdgeListReader<int,int,int> reader2("test.edges", false, false, false, false);
  auto coo2 = reader2.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo2->get_num_nnz(), 5);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Check the integrity and order of data
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
  }

  // Read it using sparsebase (weighted, directed)
  sparsebase::utils::io::EdgeListReader<int,int,float> reader3("test_values.edges", true, false, false, false);
  auto coo3 = reader3.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo3->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo3->get_vals(), nullptr);
  EXPECT_NE(coo3->get_row(), nullptr);
  EXPECT_NE(coo3->get_col(), nullptr);

  // Check the integrity and order of data
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo3->get_row()[i], row[i]);
    EXPECT_EQ(coo3->get_col()[i], col[i]);
    EXPECT_EQ(coo3->get_vals()[i], vals[i]);
  }

}

TEST(PigoMTXReader, Basics){
  // Write the mtx data to a file
  std::ofstream ofs("test_pigo.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values_pigo.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  sparsebase::utils::io::PigoMTXReader<int,int,int> reader("test_pigo.mtx", false);
  auto coo = reader.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  sparsebase::utils::io::PigoMTXReader<int,int,float> reader2("test_values_pigo.mtx", true);
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
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals[i]);
  }

}

TEST(PigoEdgeListReader, Basics){
  // Write the edge list data to a file
  std::ofstream ofs("test_pigo.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values_pigo.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  sparsebase::utils::io::PigoEdgeListReader<int,int,int> reader("test_pigo.edges");
  auto coo = reader.ReadCOO();

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo->get_num_nnz(), 5);
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo->get_row()[i], row[i]);
    EXPECT_EQ(coo->get_col()[i], col[i]);
  }

  sparsebase::utils::io::PigoEdgeListReader<int,int,float> reader2("test_values_pigo.edges", true);
  auto coo2 = reader2.ReadCOO();

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo2->get_num_nnz(), 5);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);
  EXPECT_NE(coo2->get_vals(), nullptr);

  // Check the integrity and order of data
  for(int i=0; i<5; i++){
    EXPECT_EQ(coo2->get_row()[i], row[i]);
    EXPECT_EQ(coo2->get_col()[i], col[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals[i]);
  }

}

TEST(BinaryOrderOneReader, Basics){

  // Initialize an array for testing
  int array[5]{1,2,3,4,5};
  sparsebase::format::Array<int> sbArray(5, array, sparsebase::format::kNotOwned);

  // Write the array to a binary file using sparsebase
  sparsebase::utils::io::BinaryWriterOrderOne<int> writerOrderOne("test_order_one.bin");
  writerOrderOne.WriteArray(&sbArray);

  // Read the array from the binary file using sparsebase
  sparsebase::utils::io::BinaryReaderOrderOne<int> readerOrderOne("test_order_one.bin");
  auto array2 = readerOrderOne.ReadArray();

  // Compare the arrays
  for(int i=0; i<5; i++){
    EXPECT_EQ(array[i], array2->get_vals()[i]);
  }
}

TEST(BinaryOrderTwoReader, COO){

  // Initialize a COO for testing
  int row[4]{1,2,3,4};
  int col[4]{5,6,7,8};
  float vals[4]{0.1,0.2,0.3,0.4};
  sparsebase::format::COO<int,int,float> coo(4,4,4,row,col,vals,sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int,int,float> writerOrderTwo("reader_test_order_two_coo.bin");
  writerOrderTwo.WriteCOO(&coo);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int,int,float> readerOrderTwo("reader_test_order_two_coo.bin");
  auto coo2 = readerOrderTwo.ReadCOO();

  // Compare the dimensions
  EXPECT_EQ(coo.get_dimensions(), coo2->get_dimensions());
  EXPECT_EQ(coo.get_num_nnz(), coo2->get_num_nnz());

  // Compare the underlying arrays
  for(int i=0; i<4; i++){
    EXPECT_EQ(coo.get_row()[i], coo2->get_row()[i]);
    EXPECT_EQ(coo.get_col()[i], coo2->get_col()[i]);
    EXPECT_EQ(coo.get_vals()[i], coo2->get_vals()[i]);
  }
}

TEST(BinaryOrderTwoReader, CSR) {
  // Initialize a CSR for testing
  int row_ptr[5]{0,2,3,3,4};
  int col[4]{0,2,1,3};
  float vals[4]{0.1,0.2,0.3,0.4};

  sparsebase::format::CSR<int,int,float> csr(4,4,row_ptr,col,vals,sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int,int,float> writerOrderTwo("reader_test_order_two_csr.bin");
  writerOrderTwo.WriteCSR(&csr);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int,int,float> readerOrderTwo("reader_test_order_two_csr.bin");
  auto csr2 = readerOrderTwo.ReadCSR();

  // Compare the dimensions
  EXPECT_EQ(csr.get_dimensions(), csr2->get_dimensions());
  EXPECT_EQ(csr.get_num_nnz(), csr2->get_num_nnz());

  // Compare the underlying arrays
  for(int i=0; i<4; i++){
    EXPECT_EQ(csr.get_col()[i], csr2->get_col()[i]);
    EXPECT_EQ(csr.get_vals()[i], csr2->get_vals()[i]);
  }

  for(int i=0; i<5; i++){
    EXPECT_EQ(csr.get_row_ptr()[i], csr2->get_row_ptr()[i]);
  }
}
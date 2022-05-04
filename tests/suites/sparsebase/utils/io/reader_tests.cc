#include "gtest/gtest.h"
#include <string>
#include <fstream>
#include "sparsebase/sparsebase.h"

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
  auto coo = reader.ReadCOO()->As<sparsebase::format::COO<int,int,int>>();

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 5);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Read the weighted file using sparsebase
  sparsebase::utils::io::MTXReader<int,int,float> reader2("test_values.mtx", true);
  auto coo2 = reader2.ReadCOO()->As<sparsebase::format::COO<int,int,float>>();

  // Check the dimensions
  EXPECT_EQ(coo2->get_dimensions()[0], 5);
  EXPECT_EQ(coo2->get_dimensions()[1], 5);
  EXPECT_EQ(coo2->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo2->get_vals(), nullptr);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);


  // Since COO may be sorted during construction the order of values is not certain
  std::vector<float> values(coo2->get_vals(), coo2->get_vals()+5);
  std::sort(values.begin(), values.end());

  // But they should match the values in the file otherwise
  std::vector<float> expected_values{0.1, 0.2, 0.3, 0.4, 0.5};
  for(int i=0; i<5; i++){
    EXPECT_EQ(values[i], expected_values[i]);
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
  auto coo = reader1.ReadCOO()->As<sparsebase::format::COO<int,int,int>>();

  // Check the dimensions (double the edges due to undirected read)
  EXPECT_EQ(coo->get_num_nnz(), 10);
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Read it using sparsebase (directed)
  sparsebase::utils::io::EdgeListReader<int,int,int> reader2("test.edges", false, false, false, false);
  auto coo2 = reader2.ReadCOO()->As<sparsebase::format::COO<int,int,int>>();

  // Check the dimensions
  EXPECT_EQ(coo2->get_num_nnz(), 5);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Read it using sparsebase (weighted, directed)
  sparsebase::utils::io::EdgeListReader<int,int,float> reader3("test_values.edges", true, false, false, false);
  auto coo3 = reader3.ReadCOO()->As<sparsebase::format::COO<int,int,float>>();

  // Check the dimensions
  EXPECT_EQ(coo3->get_num_nnz(), 5);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo3->get_vals(), nullptr);
  EXPECT_NE(coo3->get_row(), nullptr);
  EXPECT_NE(coo3->get_col(), nullptr);

  // Since COO may be sorted during construction the order of values is not certain
  std::vector<float> values(coo3->get_vals(), coo3->get_vals()+5);
  std::sort(values.begin(), values.end());

  // But they should match the values in the file otherwise
  std::vector<float> expected_values{0.1, 0.2, 0.3, 0.4, 0.5};
  for(int i=0; i<5; i++){
    EXPECT_EQ(values[i], expected_values[i]);
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
  int vals[4]{9,10,11,12};
  sparsebase::format::COO<int,int,int> coo(4,4,4,row,col,vals,sparsebase::format::kNotOwned);

  // Write the COO to a binary file with sparsebase
  sparsebase::utils::io::BinaryWriterOrderTwo<int,int,int> writerOrderTwo("test_order_two.bin");
  writerOrderTwo.WriteCOO(&coo);

  // Read the COO from the binary file with sparsebase
  sparsebase::utils::io::BinaryReaderOrderTwo<int,int,int> readerOrderTwo("test_order_two.bin");
  auto coo2 = readerOrderTwo.ReadCOO();

  // Compare the underlying arrays
  for(int i=0; i<4; i++){
    EXPECT_EQ(coo.get_row()[i], coo2->get_row()[i]);
    EXPECT_EQ(coo.get_col()[i], coo2->get_col()[i]);
    EXPECT_EQ(coo.get_vals()[i], coo2->get_vals()[i]);
  }
}
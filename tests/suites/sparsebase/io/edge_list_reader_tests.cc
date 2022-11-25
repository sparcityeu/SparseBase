#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
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
  sparsebase::io::EdgeListReader<int, int, int> reader1("test.edges");
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
  sparsebase::io::EdgeListReader<int, int, int> reader2("test.edges", false,
                                                        false, false, false);
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
  sparsebase::io::EdgeListReader<int, int, float> reader3(
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

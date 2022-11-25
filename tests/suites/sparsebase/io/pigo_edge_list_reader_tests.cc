#include <iostream>
#include <string>
#include <unordered_set>

#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
TEST(PigoEdgeListReader, Basics) {
  // Write the edge list data to a file
  std::ofstream ofs("test_pigo.edges");
  ofs << edge_list_data;
  ofs.close();

  // Write the edge list data with values to a file
  std::ofstream ofs2("test_values_pigo.edges");
  ofs2 << edge_list_data_with_values;
  ofs2.close();

  sparsebase::io::PigoEdgeListReader<int, int, int> reader("test_pigo.edges",
                                                           false);
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

  sparsebase::io::PigoEdgeListReader<int, int, float> reader2(
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

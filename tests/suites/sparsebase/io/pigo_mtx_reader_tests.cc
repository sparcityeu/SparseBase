#include <iostream>
#include <string>
#include <unordered_set>

#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"

TEST(PigoMTXReader, CSR) {
  // Write the mtx data to a file
  std::ofstream ofs("test_pigo.mtx");
  ofs << mtx_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values_pigo.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  sparsebase::io::PigoMTXReader<int, int, int> reader("test_pigo.mtx", false);
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

  sparsebase::io::PigoMTXReader<int, int, float> reader2("test_values_pigo.mtx",
                                                         true);
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

  sparsebase::io::PigoMTXReader<int, int, int> reader("test_pigo.mtx", false);
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

  sparsebase::io::PigoMTXReader<int, int, float> reader2("test_values_pigo.mtx",
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

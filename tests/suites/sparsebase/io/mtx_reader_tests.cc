#include <string>
#include <unordered_set>

#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"

void checkArrayReading(std::string filename) {
  // Read one column file
  sparsebase::io::MTXReader<int, int, float> reader(filename, true);
  auto array = reader.ReadArray();

  // Check the dimensions
  EXPECT_EQ(array->get_dimensions()[0], one_row_one_col_length);
  EXPECT_EQ(array->get_num_nnz(), one_row_one_col_length);
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
TEST(MTXReader, ReadingWeightedIntoVoidValues) {
  // Write the mtx data with values to a file
  std::ofstream ofs2("test_values.mtx");
  ofs2 << mtx_data_with_values;
  ofs2.close();

  // Read the weighted file using sparsebase
  EXPECT_THROW(
      (sparsebase::io::MTXReader<int, int, void>("test_values.mtx", true)),
      sparsebase::utils::ReaderException);
}
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
  sparsebase::io::MTXReader<int, int, int> reader("test.mtx");
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
  sparsebase::io::MTXReader<int, int, float> reader2("test_values.mtx", true);
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
  sparsebase::io::MTXReader<int, int, float> reader("test_arr.mtx");
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
  sparsebase::io::MTXReader<int, int, float> reader2("test_arr_values.mtx", true);
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
  sparsebase::io::MTXReader<int, int, int> reader("test_symm.mtx");
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
  sparsebase::io::MTXReader<int, int, float> reader2("test_symm_values.mtx",
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

TEST(MTXReader, BasicsSkewSymmetric) {
  // Write the mtx data to a file
  std::ofstream ofs("test_skew_symm.mtx");
  ofs << mtx_skew_symm_data;
  ofs.close();

  // Write the mtx data with values to a file
  std::ofstream ofs2("test_skew_symm_values.mtx");
  ofs2 << mtx_skew_symm_data_with_values;
  ofs2.close();

  // Read non-weighted file using sparsebase
  sparsebase::io::MTXReader<int, int, int> reader("test_skew_symm.mtx");
  auto coo = reader.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo->get_dimensions()[0], 5);
  EXPECT_EQ(coo->get_dimensions()[1], 5);
  EXPECT_EQ(coo->get_num_nnz(), 10);

  // Check that the arrays are populated
  EXPECT_NE(coo->get_row(), nullptr);
  EXPECT_NE(coo->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(coo->get_row()[i], row_skew_symm[i]);
    EXPECT_EQ(coo->get_col()[i], col_skew_symm[i]);
  }

  // Read the weighted file using sparsebase
  sparsebase::io::MTXReader<int, int, float> reader2("test_skew_symm_values.mtx",
                                                     true);
  auto coo2 = reader2.ReadCOO();

  // Check the dimensions
  EXPECT_EQ(coo2->get_dimensions()[0], 5);
  EXPECT_EQ(coo2->get_dimensions()[1], 5);
  EXPECT_EQ(coo2->get_num_nnz(), 10);

  // vals array should not be empty or null (same for the other arrays)
  EXPECT_NE(coo2->get_vals(), nullptr);
  EXPECT_NE(coo2->get_row(), nullptr);
  EXPECT_NE(coo2->get_col(), nullptr);

  // Check the integrity and order of data
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(coo2->get_row()[i], row_skew_symm[i]);
    EXPECT_EQ(coo2->get_col()[i], col_skew_symm[i]);
    EXPECT_EQ(coo2->get_vals()[i], vals_skew_symm[i]);
  }
}

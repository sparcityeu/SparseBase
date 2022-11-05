#include <string>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
TEST(BinaryOrderTwoWriter, COO) {
// Initialize a COO for testing
int row[4]{1, 2, 3, 4};
int col[4]{5, 6, 7, 8};
float vals[4]{0.1, 0.2, 0.3, 0.4};
sparsebase::format::COO<int, int, float> coo(4, 4, 4, row, col, vals,
                                             sparsebase::format::kNotOwned);

// Write the COO to a binary file with sparsebase
sparsebase::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
    "writer_test_order_two_coo.bin");
writerOrderTwo.WriteCOO(&coo);

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

TEST(BinaryOrderTwoWriter, CSR) {
// Initialize a CSR for testing
int row_ptr[5]{0, 2, 3, 3, 4};
int col[4]{0, 2, 1, 3};
float vals[4]{0.1, 0.2, 0.3, 0.4};

sparsebase::format::CSR<int, int, float> csr(4, 4, row_ptr, col, vals,
                                             sparsebase::format::kNotOwned);

// Write the COO to a binary file with sparsebase
sparsebase::io::BinaryWriterOrderTwo<int, int, float> writerOrderTwo(
    "writer_test_order_two_csr.bin");
writerOrderTwo.WriteCSR(&csr);

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

#include <iostream>
#include <string>
#include <unordered_set>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
#include "reader_data.inc"
TEST(BinaryOrderOneReader, Basics) {
  // Initialize an array for testing
  int array[5]{1, 2, 3, 4, 5};
  sparsebase::format::Array<int> sbArray(5, array,
                                         sparsebase::format::kNotOwned);

  // Write the array to a binary file using sparsebase
  sparsebase::io::BinaryWriterOrderOne<int> writerOrderOne(
      "test_order_one.bin");
  writerOrderOne.WriteArray(&sbArray);

  // Read the array from the binary file using sparsebase
  sparsebase::io::BinaryReaderOrderOne<int> readerOrderOne(
      "test_order_one.bin");
  auto array2 = readerOrderOne.ReadArray();

  // Compare the arrays
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(array[i], array2->get_vals()[i]);
  }
}

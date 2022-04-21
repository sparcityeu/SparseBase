#include "gtest/gtest.h"
#include <string>
#include "sparsebase/sparsebase.h"

TEST(BinaryOrderOneWriter, Basics){

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

TEST(BinaryOrderTwoWriter, COO){

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
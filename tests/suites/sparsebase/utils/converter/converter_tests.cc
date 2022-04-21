#include "sparsebase/sparsebase.h"
#include "gtest/gtest.h"

// The arrays defined here are for two matrices
// One in csr format one in coo format
// These are known to be equivalent (converted using scipy)
int coo_row[4]{0,0,1,3};
int coo_col[4]{0,2,1,3};
int coo_vals[4]{4,5,7,9};
int csr_row_ptr[5]{0,2,3,3,4};
int csr_col[4]{0,2,1,3};
int csr_vals[4]{4,5,7,9};


TEST(ConverterOrderTwo, CSRToCOO){
  sparsebase::format::CSR<int,int,int> csr(4,4,csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int,int,int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;
  auto coo = converterOrderTwo.Convert<sparsebase::format::COO<int,int,int>>(&csr, &cpu_context, false);

  for(int i=0; i<4; i++){
    EXPECT_EQ(coo->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo->get_vals()[i], coo_vals[i]);
  }

  auto coo2 = converterOrderTwo.Convert<sparsebase::format::COO<int,int,int>>(&csr, &cpu_context, true);

  for(int i=0; i<4; i++){
    EXPECT_EQ(coo2->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo2->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo2->get_vals()[i], coo_vals[i]);
  }
}

TEST(ConverterOrderTwo, COOToCSR){
  sparsebase::format::COO<int,int,int> coo(4,4,4,coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int,int,int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;
  auto csr = converterOrderTwo.Convert<sparsebase::format::CSR<int,int,int>>(&coo, &cpu_context, false);

  for(int i=0; i<4; i++){
    EXPECT_EQ(csr->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr->get_vals()[i], csr_vals[i]);
  }

  for(int i=0; i<5; i++){
    EXPECT_EQ(csr->get_row_ptr()[i], csr_row_ptr[i]);
  }

  auto csr2 = converterOrderTwo.Convert<sparsebase::format::CSR<int,int,int>>(&coo, &cpu_context, true);

  for(int i=0; i<4; i++){
    EXPECT_EQ(csr2->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr2->get_vals()[i], csr_vals[i]);
  }

  for(int i=0; i<5; i++){
    EXPECT_EQ(csr2->get_row_ptr()[i], csr_row_ptr[i]);
  }

}

TEST(ConverterOrderTwo, COOToCOO){
  sparsebase::format::COO<int,int,int> coo(4,4,4,coo_row, coo_col, coo_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int,int,int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;
  auto coo2 = converterOrderTwo.Convert<sparsebase::format::COO<int,int,int>>(&coo, &cpu_context);

  for(int i=0; i<4; i++){
    EXPECT_EQ(coo2->get_row()[i], coo_row[i]);
    EXPECT_EQ(coo2->get_col()[i], coo_col[i]);
    EXPECT_EQ(coo2->get_vals()[i], coo_vals[i]);
  }
}

TEST(ConverterOrderTwo, CSRToCSR){
  sparsebase::format::CSR<int,int,int> csr(4,4,csr_row_ptr, csr_col, csr_vals, sparsebase::format::kNotOwned);
  sparsebase::utils::converter::ConverterOrderTwo<int,int,int> converterOrderTwo;
  sparsebase::context::CPUContext cpu_context;
  auto csr2 = converterOrderTwo.Convert<sparsebase::format::CSR<int,int,int>>(&csr, &cpu_context);

  for(int i=0; i<4; i++){
    EXPECT_EQ(csr2->get_col()[i], csr_col[i]);
    EXPECT_EQ(csr2->get_vals()[i], csr_vals[i]);
  }

  for(int i=0; i<5; i++){
    EXPECT_EQ(csr2->get_row_ptr()[i], csr_row_ptr[i]);
  }
}
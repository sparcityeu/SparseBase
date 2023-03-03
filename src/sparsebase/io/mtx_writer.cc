#include "sparsebase/io/mtx_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
MTXWriter<IDType, NNZType, ValueType>::MTXWriter(
    std::string filename)
    : filename_(filename) {}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {
      std::ofstream mtxFile;
      mtxFile.open(filename_);
    
      //write header line
      //TODO: take header values from user
      //%MatrixMarket object format field symmetry
      std::string headerLine = "%%MatrixMarket matrix coordinate real general\n";
      mtxFile << headerLine;

      //write comment lines

      //write size line
      auto dimensions = coo->get_dimensions();
      mtxFile << dimensions[0] << " " << dimensions[1] << " " << coo->get_num_nnz() << "\n";
      
      //write data lines
      if constexpr (std::is_same_v<ValueType, void>)
      {
        throw utils::WriterException("Cannot write an MTX with void ValueType");
      }
      else{
        IDType* row = coo->get_row();
        IDType* col = coo->get_col();
        ValueType* val = coo->get_vals();
        for (int i = 0; i < coo->get_num_nnz(); i++)
        {
          mtxFile << row[i] << " " << col[i] << " " << val[i] << "\n";
        }
      }
      mtxFile.close();
      //TODO: write tests for mtx writer
}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) const {
  std::ofstream mtxFile;
      mtxFile.open(filename_);
    
      //write header line
      //TODO: take header values from user
      //%MatrixMarket object format field symmetry
      std::string headerLine = "%%MatrixMarket matrix coordinate real general\n";
      mtxFile << headerLine;

      //write comment lines

      //write size line
      auto dimensions = csr->get_dimensions();
      mtxFile << dimensions[0] << " " << dimensions[1] << " " << csr->get_num_nnz() << "\n";
      
      //write data lines
      if constexpr (std::is_same_v<ValueType, void>)
      {
        throw utils::WriterException("Cannot write an MTX with void ValueType");
        
      }
      else {
        NNZType* row = csr->get_row_ptr();
        IDType* col = csr->get_col();
        ValueType* val = csr->get_vals();
        for (int i = 0; i < csr->get_num_nnz(); i++)
        {
          mtxFile << row[i] << " " << col[i] << " " << val[i] << "\n";
        }
      }
      mtxFile.close();
}


#ifndef _HEADER_ONLY
#include "init/mtx_writer.inc"
#endif
}  // namespace sparsebase::io
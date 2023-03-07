#include "sparsebase/io/mtx_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
MTXWriter<IDType, NNZType, ValueType>::MTXWriter(
    std::string filename, std::string format, std::string field, std::string symmetry)
    : filename_(filename),
      format_(format),
      field_(field),
      symmetry_(symmetry) {}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {
      
      if constexpr (std::is_same_v<ValueType, void>)
      {
        throw utils::WriterException("Cannot write an MTX with void ValueType");
      }
      if (format_ != "array" || format_ != "coordinate") 
      {
        throw utils::ReaderException("Illegal value for the 'format' option in matrix market header");
      }
      if (field_ != "real" || field_ != "double" || field_ != "complex" || field_ != "integer" || field_ != "pattern") 
      {
        throw utils::ReaderException("Illegal value for the 'format' option in matrix market header");
      }
      if (symmetry_ != "general" || symmetry_ != "symmetric" || symmetry_ != "skew-symmetric" || symmetry_ != "hermitian") 
      {
        throw utils::ReaderException("Illegal value for the 'format' option in matrix market header");
      }
      if (format_ == "array" && field_ == "pattern") 
      {
        throw utils::ReaderException("Matrix market files with array format cannot have the field ""'pattern' ");
      }
      
      std::ofstream mtxFile;
      mtxFile.open(filename_);

      //header
      std::string headerLine = "%%MatrixMarket matrix " + format_ + " " + field_ + " " + symmetry_ + "\n";
      mtxFile << headerLine;

      //dimensions
      auto dimensions = coo->get_dimensions();
      if (format_ == "array")
      {
        mtxFile << dimensions[0] << " " << dimensions[1] << "\n";
      }
      else //cordinate
      {
        mtxFile << dimensions[0] << " " << dimensions[1] << " " << coo->get_num_nnz() << "\n";
      }
      
      else if constexpr (std::is_same_v<ValueType, int> && field_ != "integer")
      {
        //TODO add warning
      }
      else if constexpr (std::is_same_v<ValueType, float> && field_ != "real") //TODO  find out what to use for value comparison and fix it
      {
        //TODO add warning
      }
      else
      {
        IDType* row = coo->get_row();
        IDType* col = coo->get_col();
        ValueType* val = coo->get_vals();
        if (val != NULL && field_ == "pattern")
        {
          //TODO add warning, pattern selected but values given
        }


        //TODO handle symmetry
        //data lines
        for (int i = 0; i < coo->get_num_nnz(); i++)
        {
          if (format_ == "array") 
          {//format: value
              mtxFile << val[i] << "\n";
          }
          else //cordinate
          {
            if (field_ == "pattern")
            {//format: i j
              mtxFile << row[i] << " " << col[i] << "\n";
            }
            else
            {//format: i j value
              mtxFile << row[i] << " " << col[i] << " " << val[i] << "\n";
            }
          }
        }
      }
      mtxFile.close();
      //TODO: write more tests for mtx writer
}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) const {
  std::ofstream mtxFile;
      mtxFile.open(filename_);
      //TODO: after WriteCOO is finished integrate that updated version to writeCSR
      //write header line
      std::string headerLine = "%%MatrixMarket matrix " + format_ + " " + field_ + " " + symmetry_ + "\n";
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
#include "sparsebase/io/mtx_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
MTXWriter<IDType, NNZType, ValueType>::MTXWriter(
    std::string filename, std::string object, std::string format, std::string field, std::string symmetry)
    : filename_(filename),
      object_(object),
      format_(format),
      field_(field),
      symmetry_(symmetry) {}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {
      
      if (object_ != "matrix" && object_ != "vector") 
      {
        throw utils::ReaderException("Illegal value for the 'object' option in matrix market header");
      }
      else if (object_ == "vector")
      {
        throw utils::ReaderException("Matrix market writer does not currently support writing vectors.");
      }
      if (format_ != "array" && format_ != "coordinate") 
      {
        throw utils::ReaderException("Illegal value for the 'format' option in matrix market header");
      }
      if (field_ != "real" && field_ != "double" && field_ != "complex" && field_ != "integer" && field_ != "pattern") 
      {
        throw utils::ReaderException("Illegal value for the 'format' option in matrix market header");
      }
      if (symmetry_ != "general" && symmetry_ != "symmetric" && symmetry_ != "skew-symmetric" && symmetry_ != "hermitian") 
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
      std::string headerLine = "%%MatrixMarket " + object_ + " " + format_ + " " + field_ + " " + symmetry_ + "\n";
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
      
      IDType* row = coo->get_row();
      IDType* col = coo->get_col();
      ValueType* val = coo->get_vals();
      
      //TODO add warning when given field and actual data format is not same (integer vs float etc)

      if (val != NULL && field_ == "pattern")
      {
        //TODO add warning, pattern selected but values given
      }

      //data lines
      if constexpr (std::is_same_v<ValueType, void>)
      {
        throw utils::WriterException("Cannot write an MTX with void ValueType");
      }
      else
      {

        //TODO handle symmetry
        if (symmetry_ == "general")
        {
          if (format_ == "coordinate")
          {
            for (int i = 0; i < coo->get_num_nnz(); i++)
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
          else if (format_ == "array")
          {
            int index = 0;
            int current_index = 0;
            for (int i = 0; i < coo->get_num_nnz(); i++)
            {
              current_index = row[i] * dimensions[0] + col[i];
              for (int k = index; k < current_index; k++)
              {
                mtxFile << 0 << "\n";
              }
              mtxFile << val[i] << "\n";
              index = current_index + 1;
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
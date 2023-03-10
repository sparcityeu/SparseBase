#include "sparsebase/io/mtx_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csr.h"

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
      
      //illegal parameter checks
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
        throw utils::ReaderException("Illegal value for the 'field' option in matrix market header");
      }
      if (symmetry_ != "general" && symmetry_ != "symmetric" && symmetry_ != "skew-symmetric" && symmetry_ != "hermitian") 
      {
        throw utils::ReaderException("Illegal value for the 'symmetry' option in matrix market header");
      }
      if (format_ == "array" && field_ == "pattern") 
      {
        throw utils::ReaderException("Matrix market files with array format cannot have the field ""'pattern' ");
      }
      if (format_ == "array" && symmetry_ != "general") 
      {
        throw utils::ReaderException("Matrix market files with array format cannot have the property 'symmetry' ");
      }
      if (format_ == "array") 
      {
        throw utils::ReaderException("Matrix market reader does not currently support writing arrays.");
      }

      std::ofstream mtxFile;
      mtxFile.open(filename_);

      //header
      mtxFile << "%%MatrixMarket " << object_ << " " << format_ << " " << field_ << " " << symmetry_ << "\n";

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
      
      //TODO add warning when given field and actual data format is not same (integer vs float etc)

      //data lines
      if constexpr (std::is_same_v<ValueType, void>)
      {
        throw utils::WriterException("Cannot write an MTX with void ValueType");
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

        //Symmetry check
        bool saidSymmetric = (symmetry_ == "symmetric" || symmetry_ == "skew-symmetric" || symmetry_ == "hermitian");
        bool isSymmetric = false;
        
        std::vector<IDType> upper_rows(coo->get_num_nnz(), 0);
        std::vector<IDType> upper_cols(coo->get_num_nnz(), 0);
        std::vector<ValueType> upper_vals(coo->get_num_nnz());
        
        // check if number of rows and columns are equal
        if (dimensions[0] == dimensions[1]) {
          //store upper triangle
          for (int i = 0; i < coo->get_num_nnz(); i++)
          {
            if (row[i] < col[i]) //upper
            {
              upper_rows[i] = row[i];
              upper_cols[i] = col[i];
              upper_vals[i] = val[i];
            }
          }
          //check lower triangle with upper triangle
          for (int i = 0; i < coo->get_num_nnz(); i++)
          {
            if (row[i] > col[i]) //lower
            {
              bool found_symmetric = false;
              for (int j = 0; j < coo->get_num_nnz(); j++)
              {
                if (upper_rows[i] == col[i] && upper_cols[i] == row[i] && upper_vals[i] == val[i])
                {
                  found_symmetric = true;
                }
              }
              if (found_symmetric == false)
              {
                isSymmetric = false;
                break;
              }
            }
          }
        }

        //write data lines
        if (format_ == "array")
        {
          //TODO: implement writing as array format
        }
        else //coordinate
        {
          if (saidSymmetric == true && isSymmetric == false)
          {
            throw utils::ReaderException("wrong value for the 'symmetry' option in matrix market header, Matrix is not symmetric");
          }

          if (saidSymmetric)
          {
            for (int i = 0; i < coo->get_num_nnz(); i++)
            {
              if (symmetry_ != "skew-symmetric" && col[i] == val[i])
              {//on diagonal entries
                mtxFile << row[i]+1 << " " << col[i]+1;
                if (field_ == "pattern")
                {
                  mtxFile << "\n";
                }
                else
                {
                  mtxFile << " " << val[i] << "\n";
                }
              }
              if (col[i] < val[i])
              {//strictly below diagonal entries
                mtxFile << row[i]+1 << " " << col[i]+1;
                if (field_ == "pattern")
                {
                  mtxFile << "\n";
                }
                else
                {
                  mtxFile << " " << val[i] << "\n";
                }
              }
            }       
          }
          else //general
          { 
            for (int i = 0; i < coo->get_num_nnz(); i++)
            {
              mtxFile << row[i]+1 << " " << col[i]+1;
              if (field_ == "pattern")
              {
                mtxFile << "\n";
              }
              else
              {
                mtxFile << " " << val[i] << "\n";
              }
            }
          }
        }        
      }
      mtxFile.close();
}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) const {

      converter::ConverterOrderTwo<IDType, NNZType, ValueType> converterObj;
      context::CPUContext cpu_context;
      //auto coo = converterObj.template Convert<format::COO<IDType, NNZType, ValueType>>(
      //csr, &cpu_context);
      auto coo = converterObj.template Convert<format::COO<IDType, NNZType, ValueType>>(
      csr, csr->get_context(), true);
      
      WriteCOO(coo);
      
      //format::COO<IDType, NNZType, ValueType> *coo = ReadCOO();
      //return converter.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      //coo, coo->get_context(), true);
}


#ifndef _HEADER_ONLY
#include "init/mtx_writer.inc"
#endif
}  // namespace sparsebase::io
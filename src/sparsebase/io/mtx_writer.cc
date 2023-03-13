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
      if (symmetry_ == "hermitian") 
      {
        throw utils::ReaderException("Matrix market writer does not currently support hermitian symmetry.");
      }

      std::ofstream mtxFile;
      mtxFile.open(filename_);
      
      //header
      mtxFile << "%%MatrixMarket " << object_ << " " << format_ << " " << field_ << " " << symmetry_ << "\n";
      

      //TODO add warning when given field and actual data format is not same (integer vs float etc)

      if constexpr (std::is_same_v<ValueType, void>)
      {
        throw utils::WriterException("Cannot write an MTX with void ValueType");
      }
      else
      {
        auto dimensions = coo->get_dimensions();
        IDType* row = coo->get_row();
        IDType* col = coo->get_col();
        ValueType* val = coo->get_vals();

        if (val != NULL && field_ == "pattern")
        {
          std::cerr << "Warning: Pattern selected but values given.\n";
        }

        //Symmetry check
        bool saidSymmetric = (symmetry_ == "symmetric" || symmetry_ == "skew-symmetric" || symmetry_ == "hermitian");
        int NNZ = coo->get_num_nnz();
        int count_symmetric = 0;
        int count_diagonal = 0;
        if (saidSymmetric && dimensions[0] == dimensions[1]) {
            for (int i = 0; i < coo->get_num_nnz(); ++i) {
                if (row[i] != col[i]) { // Non-diagonal entry, check symmetric counterpart
                    bool found_symmetric = false;
                    for (int j = 0; j < coo->get_num_nnz(); ++j) {
                      if (symmetry_ == "skew-symmetric")
                      {
                        if (row[j] == col[i] && col[j] == row[i] && val[j] == -val[i]) {
                            found_symmetric = true;
                            count_symmetric++;
                            break;
                        }
                      }
                      else
                      {
                        if (row[j] == col[i] && col[j] == row[i] && val[j] == val[i]) {
                            found_symmetric = true;
                            count_symmetric++;
                            break;
                        }
                      }
                        
                    }
                    if (!found_symmetric) {
                        throw utils::ReaderException("Matrix is not symmetric!");
                    }
                }
                else //diagonal
                {
                  count_diagonal++;
                }
            }
            if(symmetry_ == "skew-symmetric")
            {
              NNZ -= (count_symmetric/2) + count_diagonal;
            }
            else
            {
              NNZ -= (count_symmetric/2);
            }
            
        }
      
        //dimensions and nnz
        if (format_ == "array")
        {
          mtxFile << dimensions[0] << " " << dimensions[1] << "\n";
        }
        else //cordinate
        {
          mtxFile << dimensions[0] << " " << dimensions[1] << " " << NNZ << "\n";
        }

        //write data lines
        if (format_ == "array")
        {

          //sort according to column values
          std::vector<std::pair<IDType, IDType>> sort_vec;
          for (int i = 0; i < coo->get_num_nnz(); i++) {
            sort_vec.emplace_back(col[i], row[i]);
          }
          std::sort(sort_vec.begin(), sort_vec.end(),
                    [](std::pair<IDType, IDType> t1, std::pair<IDType, IDType> t2) {
                      if (t1.first == t2.first) {
                        return t1.second < t2.second;
                      }
                      return t1.first < t2.first;
                    });

          for (int i = 0; i < coo->get_num_nnz(); i++) {
            auto &t = sort_vec[i];
            col[i] = t.first;
            row[i] = t.second;
          }

          // Write the coo matrix in array format
          int index = 0;
          int current_index = 0;
          for (int i = 0; i < coo->get_num_nnz(); i++)
          { 
            current_index = col[i] * dimensions[1] + row[i];
            while (index < current_index)
            {
              mtxFile << 0 << "\n";
              index++;
            }
            mtxFile << val[i] << "\n";
            index++;
          }
          while (index < dimensions[0]*dimensions[1])
            {
              mtxFile << 0 << "\n";
              index++;
            }
        }
        else //coordinate
        {

          if (saidSymmetric)
          {
            for (int i = 0; i < coo->get_num_nnz(); i++)
            {
              if (symmetry_ != "skew-symmetric" && col[i] == row[i])
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
              if (col[i] < row[i])
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
      auto coo = converterObj.template Convert<format::COO<IDType, NNZType, ValueType>>(
      csr, &cpu_context);
      WriteCOO(coo);
}


#ifndef _HEADER_ONLY
#include "init/mtx_writer.inc"
#endif
}  // namespace sparsebase::io
#include "sparsebase/io/mtx_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/logger.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
MTXWriter<IDType, NNZType, ValueType>::MTXWriter(
    std::string filename, std::string object, std::string format, std::string field, std::string symmetry)
    : filename_(filename),
      object_(object),
      format_(format),
      field_(field),
      symmetry_(symmetry) {}


// This function does:
// 1- Take input matrix as COO
// 2- Convert it into MTX format
// 3-Write it as .mtx file
template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {
      
      

      //information about MTX (Matrix Market Format) https://math.nist.gov/MatrixMarket/formats.html and https://networkrepository.com/mtx-matrix-market-format.html
      
      //Matrix market format has some header parameters, following if conditions check these input parameters and 
      //throws exception if its wrong, or illegal combination with other parameters or its not supported with sparsebase
      if (object_ != "matrix" && object_ != "vector") 
      {
        throw utils::WriterException("Illegal value for the 'object' option in matrix market header");
      }
      else if (object_ == "vector")
      {
        throw utils::WriterException("Matrix market writer does not currently support writing vectors.");
      }
      if (format_ != "array" && format_ != "coordinate") 
      {
        throw utils::WriterException("Illegal value for the 'format' option in matrix market header");
      }
      if (field_ != "real" && field_ != "double" && field_ != "complex" && field_ != "integer" && field_ != "pattern") 
      {
        throw utils::WriterException("Illegal value for the 'field' option in matrix market header");
      }
      if (symmetry_ != "general" && symmetry_ != "symmetric" && symmetry_ != "skew-symmetric" && symmetry_ != "hermitian") 
      {
        throw utils::WriterException("Illegal value for the 'symmetry' option in matrix market header");
      }
      if (format_ == "array" && field_ == "pattern") 
      {
        throw utils::WriterException("Matrix market files with array format cannot have the field ""'pattern' ");
      }
      if (format_ == "array" && symmetry_ != "general") 
      {
        throw utils::WriterException("Matrix market files with array format cannot have the property 'symmetry' ");
      }
      if (symmetry_ == "hermitian") 
      {
        throw utils::WriterException("Matrix market writer does not currently support hermitian symmetry.");
      }

      // open a mtx file to desired filename
      std::ofstream mtxFile;
      mtxFile.open(filename_);
      
      //insert the mtx headers which are inputted from user
      mtxFile << "%%MatrixMarket " << object_ << " " << format_ << " " << field_ << " " << symmetry_ << "\n";
      
      //TODO add warning when given field and actual data format is not same (integer vs float etc)
      
      //special condition to check if value type of matrix is void or not, MTX format supports void values if it has pattern parameter
      if constexpr (std::is_same_v<ValueType, void>)
      {
        if (field_ != "pattern")
          throw utils::WriterException("Cannot write an MTX with void ValueType, unless field is pattern.");
      }

      //consruct dimensions of the matrix
      auto dimensions = coo->get_dimensions();
      IDType* row = coo->get_row();
      IDType* col = coo->get_col();
      ValueType* val = coo->get_vals();

      //when pattern is selected no need to give values
      if (val != NULL && field_ == "pattern")
      {
        utils::Logger logger(typeid(this));
        logger.Log("Pattern selected but values given.", utils::LOG_LVL_WARNING);
      }

      //check matrix Symmetry, user inputs this as parameter but we check it anyways to detect if any user error happens.
      //also if skew symmetric or hermitian symmetric is choosen we change the matrix accordingly here before writing to file.
      bool saidSymmetric = (symmetry_ == "symmetric" || symmetry_ == "skew-symmetric" || symmetry_ == "hermitian");
      int NNZ = coo->get_num_nnz();
      int count_symmetric = 0;
      int count_diagonal = 0;
      int is_diagonal_all_zero = true;
      //check symmetry
      //saidSymmetric: user said symmetric (but it might not be)
      if (saidSymmetric == true && dimensions[0] != dimensions[1])
      {
        throw utils::WriterException("Matrix is not symmetric!");
      }
      //easier symmetry check first, it needs to be square to be symmetric
      if (saidSymmetric && dimensions[0] == dimensions[1]) 
      {
          for (int i = 0; i < coo->get_num_nnz(); ++i) 
          {
            //checking every non diagonal entry and their symmetric coordinate and compare values
            if (row[i] != col[i]) 
            { // Non-diagonal entry, check symmetric counterpart
                bool found_symmetric = false;
                for (int j = 0; j < coo->get_num_nnz(); ++j) 
                {
                  if (symmetry_ == "skew-symmetric")
                  {
                    //thse is_same_v checks are to ensure function works even if value type is void becuase of pattern parameter
                    if constexpr (std::is_same_v<ValueType, void>)
                    {
                      break; //pattern cannot be skew-symmetric
                    }
                    else
                    {
                      if (row[j] == col[i] && col[j] == row[i] && val[j] == -val[i]) 
                      {
                        found_symmetric = true;
                        count_symmetric++;
                        break;
                      }
                    }
                  }
                  else
                  {
                    //thse is_same_v checks are to ensure function works even if value type is void becuase of pattern parameter
                    if constexpr (std::is_same_v<ValueType, void>)
                    {
                      if (row[j] == col[i] && col[j] == row[i]) 
                      {
                        found_symmetric = true;
                        count_symmetric++;
                        break;
                      }
                    }
                    else
                    {
                      if (row[j] == col[i] && col[j] == row[i] && val[j] == val[i]) 
                      {
                        found_symmetric = true;
                        count_symmetric++;
                        break;
                      }
                    }
                  }
                }
                //found_symmetric is true if matrix is actually symmetric (user might said not symmetric)
                if (!found_symmetric) {
                    throw utils::WriterException("Matrix is not symmetric!");
                }
            }
            else //diagonal
            {
              count_diagonal++;
              //thse is_same_v checks are to ensure function works even if value type is void becuase of pattern parameter
              if constexpr (std::is_same_v<ValueType, void>)
              {
                //zero
              }
              else
              {
                if (val[i] != 0)
                {
                  //checking this because we need this for skew-symmetry check
                  is_diagonal_all_zero = false;
                }
              }
            }
          }
          //check if matrix is actually skew-symmetric
          if(symmetry_ == "skew-symmetric")
          {
            if (!is_diagonal_all_zero)
            {
              throw utils::WriterException("Skew-symmetric matrix with non-zero diagonal values!");
            }
            NNZ -= (count_symmetric/2) + count_diagonal;
          }
          else
          {
            NNZ -= (count_symmetric/2);
          }
      }

      //dimensions and nnz
      //if array is given rather than coordinate, dimension format of the file is different, so it is handled here
      if (format_ == "array")
      {
        mtxFile << dimensions[0] << " " << dimensions[1] << "\n";
      }
      else //cordinate
      {
        mtxFile << dimensions[0] << " " << dimensions[1] << " " << NNZ << "\n";
      }
      //data lines are written here for array
      if (format_ == "array")
      {
        //write data lines
        if constexpr (std::is_same_v<ValueType, void>)
        { 
          int index = 0;
          while (index < dimensions[0]*dimensions[1])
            {
              mtxFile << 0 << "\n";
              index++;
            }
        }
        else
        {
          //sort according to column values
          std::vector<std::tuple<IDType, IDType, ValueType>> sort_vec;
          for (int i = 0; i < coo->get_num_nnz(); i++) {
            sort_vec.emplace_back(col[i], row[i], val[i]);
          }
          std::sort(sort_vec.begin(), sort_vec.end(),
                    [](std::tuple<IDType, IDType, ValueType> t1, std::tuple<IDType, IDType, ValueType> t2) {
                      if (std::get<0>(t1) == std::get<0>(t2)) {
                        return std::get<1>(t1) < std::get<1>(t2);
                      }
                      return std::get<0>(t1) < std::get<0>(t2);
                    });

          // Write the coo matrix in array format
          int index = 0;
          int current_index = 0;
          for (int i = 0; i < coo->get_num_nnz(); i++)
          { 
            current_index = std::get<0>(sort_vec[i]) * dimensions[0] + std::get<1>(sort_vec[i]);
            while (index < current_index)
            {
              mtxFile << 0 << "\n";
              index++;
            }
            mtxFile << std::get<2>(sort_vec[i]) << "\n";
            index++;
          }
          while (index < dimensions[0]*dimensions[1])
            {
              mtxFile << 0 << "\n";
              index++;
            }
        }
      }
      //data lines for coordinate are written here
      else //coordinate
      {

        if (saidSymmetric)
        {
          for (int i = 0; i < coo->get_num_nnz(); i++)
          {
            if (symmetry_ != "skew-symmetric" && col[i] == row[i])
            {//on diagonal entries
              mtxFile << row[i]+1 << " " << col[i]+1;
              //thse is_same_v checks are to ensure function works even if value type is void becuase of pattern parameter
              if constexpr (std::is_same_v<ValueType, void>)
              {
                if (field_ == "pattern")
                  mtxFile << "\n";
              }
              else
              {
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
            if (col[i] < row[i])
            {//strictly below diagonal entries
              mtxFile << row[i]+1 << " " << col[i]+1;
              //thse is_same_v checks are to ensure function works even if value type is void becuase of pattern parameter
              if constexpr (std::is_same_v<ValueType, void>)
              {
                if (field_ == "pattern")
                  mtxFile << "\n";
              } 
              else
              { 
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
        //data lines for general format is written here
        else //general
        { 
          for (int i = 0; i < coo->get_num_nnz(); i++)
          {
            mtxFile << row[i]+1 << " " << col[i]+1;
            //thse is_same_v checks are to ensure function works even if value type is void becuase of pattern parameter
            if constexpr (std::is_same_v<ValueType, void>)
            {
              if (field_ == "pattern")
                mtxFile << "\n";
            }
            else
            {
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
      //this is for writing mtx file from CSR rather than COO
      //this function doesnt convert CSR to MTX, it rather converts CSR to COO using sparsebase conversion
      //then calls above WriteCOO function with this converted COO
      converter::ConverterOrderTwo<IDType, NNZType, ValueType> converterObj;
      context::CPUContext cpu_context;
      auto coo = converterObj.template Convert<format::COO<IDType, NNZType, ValueType>>(
      csr, &cpu_context);
      WriteCOO(coo);
}

template <typename IDType, typename NNZType, typename ValueType>
void MTXWriter<IDType, NNZType, ValueType>::WriteArray(
  format::Array<ValueType> *arr) const {
    //this function takes array as input and writes it strictly as array format
    //it is same as WriteCOO otherwise    
    
    //illegal parameter checks
    if (object_ != "matrix" && object_ != "vector") 
    {
      throw utils::WriterException("Illegal value for the 'object' option in matrix market header");
    }
    else if (object_ == "vector")
    {
      throw utils::WriterException("Matrix market writer does not currently support writing vectors.");
    }
    if (format_ != "array" && format_ != "coordinate") 
    {
      throw utils::WriterException("Illegal value for the 'format' option in matrix market header");
    }
    if (field_ != "real" && field_ != "double" && field_ != "complex" && field_ != "integer" && field_ != "pattern") 
    {
      throw utils::WriterException("Illegal value for the 'field' option in matrix market header");
    }
    if (symmetry_ != "general" && symmetry_ != "symmetric" && symmetry_ != "skew-symmetric" && symmetry_ != "hermitian") 
    {
      throw utils::WriterException("Illegal value for the 'symmetry' option in matrix market header");
    }
    if (format_ == "array" && field_ == "pattern") 
    {
      throw utils::WriterException("Matrix market files with array format cannot have the field ""'pattern' ");
    }
    if (format_ == "array" && symmetry_ != "general") 
    {
      throw utils::WriterException("Matrix market files with array format cannot have the property 'symmetry' ");
    }
    if (symmetry_ == "hermitian") 
    {
      throw utils::WriterException("Matrix market writer does not currently support hermitian symmetry.");
    }
    if (format_ == "coordinate")
    {
      throw utils::WriterException("Matrix market writer does not currently support writing array as coordinate.");
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
      ValueType* val = arr->get_vals();
      auto dimensions = arr->get_dimensions();
      //dimensions
      mtxFile << 1 << " " << dimensions[0] << "\n";
      
      for (int i = 0; i < dimensions[0]; i++)
      { 
        mtxFile << val[i] << "\n";
      }
      mtxFile.close();
    }
  }

#ifndef _HEADER_ONLY
#include "init/mtx_writer.inc"
#endif
}  // namespace sparsebase::io
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
      //output stream
      std::ofstream mtxFile;
      mtxFile.open(filename_);
    
      //write header line
      std::string headerLine = "%%MatrixMarket matrix coordinate pattern general\n";
      mtxFile << headerLine;

      //write comment lines

      //write size line
      auto dimensions = coo->get_dimensions();
      mtxFile << dimensions[0] << " " << dimensions[1] << " " << coo->get_num_nnz() << "\n";
      
      //write data lines
      auto line = coo->get_vals();
      while(line != nullptr)
        if constexpr (!std::is_same_v<ValueType, void>)
          mtxFile << line[0] << line[1] << line[2] << "\n";
          line = coo->get_vals();

      mtxFile.close();
}
#ifndef _HEADER_ONLY
#include "init/mtx_writer.inc"
#endif
}  // namespace sparsebase::io
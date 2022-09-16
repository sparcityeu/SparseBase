//
// Created by Amro on 9/14/2022.
//

#ifndef SPARSEBASE_PROJECT_IOBASE_H
#define SPARSEBASE_PROJECT_IOBASE_H

#include "sparsebase/config.h"
#include "sparsebase/format/format.h"
#include "sparsebase/utils/io/reader.h"
#include "sparsebase/utils/io/writer.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace sparsebase;
namespace sparsebase{

namespace utils {
namespace io {
class IOBase {
public:
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType> * ReadMTXToCSR(std::string filename, bool convert_index_to_zero=true){
    MTXReader<IDType, NNZType, ValueType> reader(filename, convert_index_to_zero);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType> * ReadMTXToCOO(std::string filename, bool convert_index_to_zero=true){
    MTXReader<IDType, NNZType, ValueType> reader(filename, convert_index_to_zero);
    return reader.ReadCOO();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::Array<ValueType> * ReadMTXToArray(std::string filename, bool convert_index_to_zero=true){
    MTXReader<IDType, NNZType, ValueType> reader(filename, convert_index_to_zero);
    return reader.ReadArray();
  }
};
}
}
}
#endif // SPARSEBASE_PROJECT_IOBASE_H

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
 // PIGO

  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType> * ReadPigoMTXToCSR(std::string filename, bool weighted, bool convert_index_to_zero=true){
    PigoMTXReader<IDType, NNZType, ValueType> reader(filename, weighted, convert_index_to_zero);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType> * ReadPigoMTXToCOO(std::string filename, bool weighted, bool convert_index_to_zero=true){
    PigoMTXReader<IDType, NNZType, ValueType> reader(filename, weighted, convert_index_to_zero);
    return reader.ReadCOO();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType> * ReadPigoEdgeListToCSR(std::string filename, bool weighted){
    PigoEdgeListReader<IDType, NNZType, ValueType> reader(filename, weighted);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType> * ReadPigoEdgeListToCOO(std::string filename, bool weighted){
    PigoEdgeListReader<IDType, NNZType, ValueType> reader(filename, weighted);
    return reader.ReadCOO();
  }

  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType> * ReadEdgeListToCSR(std::string filename, bool weighted = false, bool remove_self_edges = false, bool read_undirected = true, bool square = false){
    EdgeListReader<IDType, NNZType, ValueType> reader(filename, weighted, true, remove_self_edges, read_undirected, square);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType> * ReadEdgeListToCOO(std::string filename, bool weighted = false, bool remove_self_edges = false, bool read_undirected = true, bool square = false){
    EdgeListReader<IDType, NNZType, ValueType> reader(filename, weighted, true, remove_self_edges, read_undirected, square);
    return reader.ReadCOO();
  }

  // Binary reader
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType> * ReadBinaryToCSR(std::string filename){
    BinaryReaderOrderTwo<IDType, NNZType, ValueType> reader(filename);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType> * ReadBinaryToCOO(std::string filename){
    BinaryReaderOrderTwo<IDType, NNZType, ValueType> reader(filename);
    return reader.ReadCOO();
  }
  template <typename ValueType>
  static format::Array<ValueType> * ReadBinaryToArray(std::string filename){
    BinaryReaderOrderOne<ValueType> reader(filename);
    return reader.ReadArray();
  }

  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteCOOToBinary(format::COO<IDType, NNZType, ValueType>* coo, std::string filename){
    BinaryWriterOrderTwo<IDType, NNZType, ValueType> writer(filename);
    return writer.WriteCOO(coo);
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteCSRToBinary(format::CSR<IDType, NNZType, ValueType>* csr, std::string filename){
    BinaryWriterOrderTwo<IDType, NNZType, ValueType> writer(filename);
    return writer.WriteCSR(csr);
  }
  template <typename ValueType>
  static void WriteArrayToBinary(format::Array<ValueType>* array, std::string filename){
    BinaryWriterOrderOne<ValueType> writer(filename);
    return writer.WriteArray(array);
  }
};
}
}
}
#endif // SPARSEBASE_PROJECT_IOBASE_H

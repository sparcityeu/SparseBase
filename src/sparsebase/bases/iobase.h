#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/io/binary_reader_order_one.h"
#include "sparsebase/io/binary_reader_order_two.h"
#include "sparsebase/io/binary_writer_order_one.h"
#include "sparsebase/io/binary_writer_order_two.h"
#include "sparsebase/io/edge_list_reader.h"
#include "sparsebase/io/mtx_reader.h"
#include "sparsebase/io/mtx_writer.h"
#include "sparsebase/io/pigo_edge_list_reader.h"
#include "sparsebase/io/pigo_mtx_reader.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/io/writer.h"

#ifndef SPARSEBASE_PROJECT_IOBASE_H
#define SPARSEBASE_PROJECT_IOBASE_H

namespace sparsebase::bases {
class IOBase {
 public:
  //! Read a matrix market file into a CSR format
  /*!
   * Read the matrix market file in the path `filename` into a `CSR<IDType,
   * NNZType, ValueType>` object.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename the path to the matrix market file.
   * @param convert_index_to_zero whether to convert indices in the matrix
   * market to zero-indexing or to read them as is and add an empty zeroth row
   * and column.
   * @return a pointer at a `CSR<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType>* ReadMTXToCSR(
      std::string filename, bool convert_index_to_zero = true) {
    io::MTXReader<IDType, NNZType, ValueType> reader(filename,
                                                     convert_index_to_zero);
    return reader.ReadCSR();
  }
  //! Read a matrix market file into a COO format
  /*!
   * Read the matrix market file in the path `filename` into a `COO<IDType,
   * NNZType, ValueType>` object.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename the path to the matrix market file.
   * @param convert_index_to_zero whether to convert indices in the matrix
   * market to zero-indexing or to read them as is and add an empty zeroth row
   * and column.
   * @return a pointer at a `COO<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType>* ReadMTXToCOO(
      std::string filename, bool convert_index_to_zero = true) {
    io::MTXReader<IDType, NNZType, ValueType> reader(filename,
                                                     convert_index_to_zero);
    return reader.ReadCOO();
  }
  //! Read a matrix market file into a dense array format.
  /*!
   * Read the matrix market file in the path `filename` into a
   * `Array<ValueType>` object. If the matrix inside the file has more than a
   * single row or more than a single column, an exception will be thrown.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename the path to the matrix market file.
   * @param convert_index_to_zero whether to convert indices in the matrix
   * market to zero-indexing or to read them as is and add an empty zeroth row
   * and column.
   * @return a pointer at a `Array<ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::Array<ValueType>* ReadMTXToArray(
      std::string filename, bool convert_index_to_zero = true) {
    io::MTXReader<IDType, NNZType, ValueType> reader(filename,
                                                     convert_index_to_zero);
    return reader.ReadArray();
  }

  //! Read a matrix market file into a CSR format using PIGO parallel reading.
  /*!
   * Read the matrix market file in the path `filename` into a `CSR<IDType,
   * NNZType, ValueType>` object.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename the path to the matrix market file.
   * @param convert_index_to_zero whether to convert indices in the matrix
   * market to zero-indexing or to read them as is and add an empty zeroth row
   * and column.
   * @return a pointer at a `CSR<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType>* ReadPigoMTXToCSR(
      std::string filename, bool weighted, bool convert_index_to_zero = true) {
    io::PigoMTXReader<IDType, NNZType, ValueType> reader(filename, weighted,
                                                         convert_index_to_zero);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType>* ReadPigoMTXToCOO(
      std::string filename, bool weighted, bool convert_index_to_zero = true) {
    io::PigoMTXReader<IDType, NNZType, ValueType> reader(filename, weighted,
                                                         convert_index_to_zero);
    return reader.ReadCOO();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType>* ReadPigoEdgeListToCSR(
      std::string filename, bool weighted) {
    io::PigoEdgeListReader<IDType, NNZType, ValueType> reader(filename,
                                                              weighted);
    return reader.ReadCSR();
  }
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType>* ReadPigoEdgeListToCOO(
      std::string filename, bool weighted) {
    io::PigoEdgeListReader<IDType, NNZType, ValueType> reader(filename,
                                                              weighted);
    return reader.ReadCOO();
  }

  //! Read an edge list file into a CSR object
  /*!
   * Reads an edge list file into a `CSR<IDType, NNZType, ValueType>` object.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename path to the edge list file.
   * @param weighted whether the edge list file contains edge weights.
   * @param remove_self_edges whether self-edges should be removed.
   * @param read_undirected if true, reading the edge (u,v) will also add the
   * edge (v,u) to the object.
   * @param square whether the graph is square.
   * @return A pointer at a `CSR<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType>* ReadEdgeListToCSR(
      std::string filename, bool weighted = false,
      bool remove_self_edges = false, bool read_undirected = true,
      bool square = false) {
    io::EdgeListReader<IDType, NNZType, ValueType> reader(
        filename, weighted, true, remove_self_edges, read_undirected, square);
    return reader.ReadCSR();
  }
  //! Read an edge list file into a COO object
  /*!
   * Reads an edge list file into a `COO<IDType, NNZType, ValueType>` object.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename path to the edge list file.
   * @param weighted whether the edge list file contains edge weights.
   * @param remove_self_edges whether self-edges should be removed.
   * @param read_undirected if true, reading the edge (u,v) will also add the
   * edge (v,u) to the object.
   * @param square whether the graph is square.
   * @return A pointer at a `COO<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType>* ReadEdgeListToCOO(
      std::string filename, bool weighted = false,
      bool remove_self_edges = false, bool read_undirected = true,
      bool square = false) {
    io::EdgeListReader<IDType, NNZType, ValueType> reader(
        filename, weighted, true, remove_self_edges, read_undirected, square);
    return reader.ReadCOO();
  }

  //! Read an SparseBase Binary file into a CSR
  /*!
   * Reads a SparseBase binary file into a `CSR<IDType, NNZType, ValueType>`
   * object
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename path to the edge list file.
   * @return A pointer at a `CSR<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::CSR<IDType, NNZType, ValueType>* ReadBinaryToCSR(
      std::string filename) {
    io::BinaryReaderOrderTwo<IDType, NNZType, ValueType> reader(filename);
    return reader.ReadCSR();
  }
  //! Read an SparseBase Binary file into a COO
  /*!
   * Reads a SparseBase binary file into a `COO<IDType, NNZType, ValueType>`
   * object
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename path to the edge list file.
   * @return A pointer at a `COO<IDType, NNZType, ValueType>` object.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static format::COO<IDType, NNZType, ValueType>* ReadBinaryToCOO(
      std::string filename) {
    io::BinaryReaderOrderTwo<IDType, NNZType, ValueType> reader(filename);
    return reader.ReadCOO();
  }

  //! Read an SparseBase Binary file into a Array
  /*!
   * Reads a SparseBase binary file into a `Array<ValueType>` object
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param filename path to the edge list file.
   * @return A pointer at a `Array<ValueType>` object.
   */
  template <typename ValueType>
  static format::Array<ValueType>* ReadBinaryToArray(std::string filename) {
    io::BinaryReaderOrderOne<ValueType> reader(filename);
    return reader.ReadArray();
  }

  //! Write a COO object to a SparseBase binary file
  /*!
   * Write a COO object to a SparseBase binary file.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param coo a pointer at the `COO<IDType, NNZType, ValueType>` object to
   * write.
   * @param filename path to write the file.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteCOOToBinary(format::COO<IDType, NNZType, ValueType>* coo,
                               std::string filename) {
    io::BinaryWriterOrderTwo<IDType, NNZType, ValueType> writer(filename);
    return writer.WriteCOO(coo);
  }
  //! Write a CSR object to a SparseBase binary file
  /*!
   * Write a CSR object to a SparseBase binary file.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param csr a pointer at the `CSR<IDType, NNZType, ValueType>` object to
   * write.
   * @param filename path to write the file.
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteCSRToBinary(format::CSR<IDType, NNZType, ValueType>* csr,
                               std::string filename) {
    io::BinaryWriterOrderTwo<IDType, NNZType, ValueType> writer(filename);
    return writer.WriteCSR(csr);
  }
  //! Write an Array object to a SparseBase binary file
  /*!
   * Write a Array object to a SparseBase binary file.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param array a pointer at the `Array<ValueType>` object to write.
   * @param filename path to write the file.
   */
  template <typename ValueType>
  static void WriteArrayToBinary(format::Array<ValueType>* array,
                                 std::string filename) {
    io::BinaryWriterOrderOne<ValueType> writer(filename);
    return writer.WriteArray(array);
  }
  //! Write a CSR object to a matrix market file
  /*!
   * Write a CSR object to a matrix market file.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param csr a pointer at the `CSR<IDType, NNZType, ValueType>` object to write.
   * @param filename path to write the file.
   * @param object is either matrix or vector.
   * @param format is either coordinate or array.
   * @param field is either real, double, complex, integer or pattern.
   * @param symmetry is either general (legal for real, complex,
    integer or pattern fields), symmetric (real, complex, integer or pattern),
     skew-symmetric (real, complex or integer), or hermitian (complex only).
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteCSRToMTX(
      format::CSR<IDType, NNZType, ValueType>* csr,
      std::string filename,
      std::string object = "matrix",
      std::string format = "coordinate",
      std::string field = "real",
      std::string symmetry = "general") {
    io::MTXWriter<IDType, NNZType, ValueType> writer(filename,
                                                     object,
                                                     format,
                                                     field,
                                                     symmetry);
    return writer.WriteCSR(csr);
  }
  //! Write a COO object to a matrix market file
  /*!
   * Write a COO object to a matrix market file.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the matrix (vertex
   * weights in the case of a graph).
   * @param coo a pointer at the `COO<IDType, NNZType, ValueType>` object to write.
   * @param filename path to write the file.
   * @param object is either matrix or vector.
   * @param format is either coordinate or array.
   * @param field is either real, double, complex, integer or pattern.
   * @param symmetry is either general (legal for real, complex,
    integer or pattern fields), symmetric (real, complex, integer or pattern),
     skew-symmetric (real, complex or integer), or hermitian (complex only).
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteCOOToMTX(
      format::CSR<IDType, NNZType, ValueType>* coo,
      std::string filename,
      std::string object = "matrix",
      std::string format = "coordinate",
      std::string field = "real",
      std::string symmetry = "general") {
    io::MTXWriter<IDType, NNZType, ValueType> writer(filename,
                                                     object,
                                                     format,
                                                     field,
                                                     symmetry);
    return writer.WriteCOO(coo);
  }
  //! Write an Array object to a matrix market file
  /*!
   * Write an Array object to a matrix market file.
   * @tparam IDType type to represent the number of rows and columns in the
   * object.
   * @tparam NNZType type to represent the number of non-zeros in the object.
   * @tparam ValueType type to represent the data inside the array (vertex
   * weights in the case of a graph).
   * @param arr a pointer at the `Array<ValueType>` object to write.
   * @param filename path to write the file.
   * @param object is either matrix or vector.
   * @param format is either coordinate or array.
   * @param field is either real, double, complex, integer or pattern.
   * @param symmetry is either general (legal for real, complex,
    integer or pattern fields), symmetric (real, complex, integer or pattern),
     skew-symmetric (real, complex or integer), or hermitian (complex only).
   */
  template <typename IDType, typename NNZType, typename ValueType>
  static void WriteArrayToMTX(
      format::Array<ValueType>* arr,
      std::string filename,
      std::string object = "matrix",
      std::string format = "coordinate",
      std::string field = "real",
      std::string symmetry = "general") {
    io::MTXWriter<IDType, NNZType, ValueType> writer(filename,
                                                     object,
                                                     format,
                                                     field,
                                                     symmetry);
    return writer.WriteArray(arr);
  }
};
}  // namespace sparsebase::bases
#endif  // SPARSEBASE_PROJECT_IOBASE_H

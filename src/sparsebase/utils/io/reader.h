/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_IO_READER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_IO_READER_H_

#include "sparsebase/config.h"
#include "sparsebase/format/format.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace sparsebase {

namespace utils {

namespace io {

//! Base class for all readers, has no special functionality on its own
class Reader {
public:
  virtual ~Reader() = default;
};

//! Interface for readers that can return a CSR instance
template <typename IDType, typename NNZType, typename ValueType>
class ReadsCSR {
public:
  //! Reads the file to a CSR instance and returns a pointer to it
  virtual format::CSR<IDType, NNZType, ValueType> *ReadCSR() const = 0;
};

//! Interface for readers that can return a COO instance
template <typename IDType, typename NNZType, typename ValueType>
class ReadsCOO {
public:
  //! Reads the file to a COO instance and returns a pointer to it
  virtual format::COO<IDType, NNZType, ValueType> *ReadCOO() const = 0;
};

//! Interface for readers that can return an Array instance
template <typename T> class ReadsArray {
public:
  //! Reads the file to an Array instance and returns a pointer to it
  virtual format::Array<T> *ReadArray() const = 0;
};

//! Reader for the Edge List file format
/*!
 * Reads files of the following format:
 * - Each line contains 2 ids (vertices for a graph, cols/rows for a matrix)
 * followed by an optional weight
 * - Delimiters should be spaces or tabs
 * - Each line represents a connection between the specified ids with the given
 * weight
 */
template <typename IDType, typename NNZType, typename ValueType>
class EdgeListReader : public Reader,
                       public ReadsCSR<IDType, NNZType, ValueType>,
                       public ReadsCOO<IDType, NNZType, ValueType> {
public:
  /*!
   * Constructor for the EdgeListReader class
   * @param filename path to the file to be read
   * @param weighted should be set to true if the file contains weights
   * @param remove_duplicates if set to true duplicate connections will be
   * removed
   * @param remove_self_edges if set to true connections from any vertex to
   * itself will be removed
   * @param read_undirected_ if set to true for any entry (u,v) both (u,v) and
   * (v,u) will be read
   */
  explicit EdgeListReader(std::string filename, bool weighted = false,
                          bool remove_duplicates = false,
                          bool remove_self_edges = false,
                          bool read_undirected_ = true, bool square = false);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  ~EdgeListReader() override;

private:
  std::string filename_;
  bool weighted_;
  bool remove_duplicates_;
  bool remove_self_edges_;
  bool read_undirected_;
  bool square_;
};

//! Reader for the Matrix Market File Format
/*!
 * Detailed explanations of the MTX format can be found in these links:
 * - https://networkrepository.com/mtx-matrix-market-format.html
 * - https://math.nist.gov/MatrixMarket/formats.html
 */
template <typename IDType, typename NNZType, typename ValueType>
class MTXReader : public Reader,
                  public ReadsCSR<IDType, NNZType, ValueType>,
                  public ReadsCOO<IDType, NNZType, ValueType>,
                  public ReadsArray<ValueType> {
public:
  /*!
   * Constructor for the MTXReader class
   * @param filename path to the file to be read
   * @param convert_to_zero_index if set to true the indices will be converted
   * such that they start from 0 instead of 1
   */
  explicit MTXReader(std::string filename, bool convert_to_zero_index = true);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;
  format::Array<ValueType> *ReadArray() const override;
  ~MTXReader() override;

private:
  enum MTXObjectOptions {
    matrix,
    vector
  };
  enum MTXFormatOptions {
    coordinate,
    array
  };
  enum MTXFieldOptions {
    real,
    double_field,
    complex,
    integer,
    pattern
  };
  enum MTXSymmetryOptions {
    general = 0,
    symmetric = 1,
    skew_symmetric = 2,
    hermitian = 3
  };
  struct MTXOptions {
    MTXObjectOptions object;
    MTXFormatOptions format;
    MTXFieldOptions field;
    MTXSymmetryOptions symmetry;
  };
  MTXOptions ParseHeader(std::string header_line) const;
  format::Array<ValueType> *ReadCoordinateIntoArray() const;
  format::Array<ValueType> *ReadArrayIntoArray() const;
  template <bool weighted>
  format::COO<IDType, NNZType, ValueType> *ReadArrayIntoCOO() const;
  template <bool weighted, int symm, bool conv_to_zero>
  format::COO<IDType, NNZType, ValueType> *ReadCoordinateIntoCOO() const;
  std::string filename_;
  bool convert_to_zero_index_;
  MTXOptions options_;
};

/*!
 * A parallelized MTX reader using the PIGO library
 * (This feature is currently experimental and not available on all platforms,
 * if you have problems please use one of the provided sequential readers)
 * More information about PIGO: https://github.com/GT-TDAlab/PIGO
 */
template <typename IDType, typename NNZType, typename ValueType>
class PigoMTXReader : public Reader,
                      public ReadsCOO<IDType, NNZType, ValueType>,
                      public ReadsCSR<IDType, NNZType, ValueType> {
public:
  PigoMTXReader(std::string filename, bool weighted,
                bool convert_to_zero_index = true);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;
  //format::Array<ValueType> *ReadArray() const override;
  virtual ~PigoMTXReader() = default;

private:
  std::string filename_;
  bool weighted_;
  bool convert_to_zero_index_;
};

/*!
 * A parallelized EdgeList reader using the PIGO library
 * (This feature is currently experimental and not available on all platforms,
 * if you have problems please use one of the provided sequential readers)
 * More information about PIGO: https://github.com/GT-TDAlab/PIGO
 */
template <typename IDType, typename NNZType, typename ValueType>
class PigoEdgeListReader : public Reader,
                           public ReadsCSR<IDType, NNZType, ValueType>,
                           public ReadsCOO<IDType, NNZType, ValueType> {
public:
  PigoEdgeListReader(std::string filename, bool weighted);
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  virtual ~PigoEdgeListReader() = default;

private:
  std::string filename_;
  bool weighted_;
};

//! Reads files encoded in SparseBase's custom binary format (CSR and COO)
template <typename IDType, typename NNZType, typename ValueType>
class BinaryReaderOrderTwo : public Reader,
                             public ReadsCSR<IDType, NNZType, ValueType>,
                             public ReadsCOO<IDType, NNZType, ValueType> {
public:
  explicit BinaryReaderOrderTwo(std::string filename);
  ~BinaryReaderOrderTwo() override = default;
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;

private:
  std::string filename_;
};

//! Reads files encoded in SparseBase's custom binary format (Array)
template <typename T>
class BinaryReaderOrderOne : public Reader, public ReadsArray<T> {
public:
  explicit BinaryReaderOrderOne(std::string filename);
  ~BinaryReaderOrderOne() override = default;
  format::Array<T> *ReadArray() const override;

private:
  std::string filename_;
};

} // namespace io

} // namespace utils

} // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparsebase/utils/io/reader.cc"
#endif
#endif // SPARSEBASE_SPARSEBASE_UTILS_IO_READER_H_

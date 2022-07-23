/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda Sener
 * All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_IO_WRITER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_IO_WRITER_H_

#include "sparsebase/format/format.h"
#include "sparsebase/utils/io/reader.h"
#include <string>

namespace sparsebase {

namespace utils {

namespace io {

//! Base class for all writers, has no special functionality on its own
class Writer {
public:
  virtual ~Writer() = default;
};

//! Interface for writers that can write a CSR instance to a file
template <typename IDType, typename NNZType, typename ValueType>
class WritesCSR {
  //! Writes the given CSR instance to a file
  virtual void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const = 0;
};

//! Interface for writers that can write a COO instance to a file
template <typename IDType, typename NNZType, typename ValueType>
class WritesCOO {
  //! Writes the given COO instance to a file
  virtual void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const = 0;
};

//! Interface for writers that can write an Array instance to a file
template <typename T> class WritesArray {
  //! Writes the given Array instance to a file
  virtual void WriteArray(format::Array<T> *arr) const = 0;
};

//! Writes files by encoding them in SparseBase's custom binary format (CSR and
//! COO)
template <typename IDType, typename NNZType, typename ValueType>
class BinaryWriterOrderTwo : public Writer,
                             public WritesCOO<IDType, NNZType, ValueType>,
                             public WritesCSR<IDType, NNZType, ValueType> {
public:
  explicit BinaryWriterOrderTwo(std::string filename);
  ~BinaryWriterOrderTwo() override = default;
  void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const override;
  void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const override;

private:
  std::string filename_;
};

//! Writes files by encoding them in SparseBase's custom binary format (Array)
template <typename T>
class BinaryWriterOrderOne : public Writer, public WritesArray<T> {
public:
  explicit BinaryWriterOrderOne(std::string filename);
  ~BinaryWriterOrderOne() override = default;
  void WriteArray(format::Array<T> *arr) const override;

private:
  std::string filename_;
};

} // namespace io

} // namespace utils

} // namespace sparsebase

#ifdef _HEADER_ONLY
#include "sparsebase/utils/io/writer.cc"
#endif

#endif // SPARSEBASE_SPARSEBASE_UTILS_IO_WRITER_H_
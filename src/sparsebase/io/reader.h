/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_IO_READER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_IO_READER_H_

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/format/higher_order_coo.h"

namespace sparsebase {

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
template <typename T>
class ReadsArray {
 public:
  //! Reads the file to an Array instance and returns a pointer to it
  virtual format::Array<T> *ReadArray() const = 0;
};

//! Interface for readers that can return a HigherOrderCOO instance
template <typename IDType, typename NNZType, typename ValueType>
class ReadsHigherOrderCOO {
 public:
  //! Reads the file to a HigherOrderCOO instance and returns a pointer to it
  virtual format::HigherOrderCOO<IDType, NNZType, ValueType> *ReadHigherOrderCOO() const = 0;
};
}  // namespace io

}  // namespace sparsebase
#endif  // SPARSEBASE_SPARSEBASE_UTILS_IO_READER_H_

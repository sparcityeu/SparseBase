/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_IO_WRITER_H_
#define SPARSEBASE_SPARSEBASE_UTILS_IO_WRITER_H_

#include <string>

#include "sparsebase/format/array.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/object/object.h"

namespace sparsebase {

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
template <typename T>
class WritesArray {
  //! Writes the given Array instance to a file
  virtual void WriteArray(format::Array<T> *arr) const = 0;
};

//! Interface for writers that can write an Graph instance to a file
template <typename IDType, typename NNZType, typename ValueType>
class WritesGraph {
  //! Writes the given Graph instance to a file
  virtual void WriteGraph(object::Graph<IDType, NNZType, ValueType> *graph) const = 0;
};

//! Interface for writers that can write an HyperGraph instance to a file
template <typename IDType, typename NNZType, typename ValueType>
class WritesHyperGraph {
  //! Writes the given HyperGraph instance to a file
  virtual void WriteHyperGraph(object::HyperGraph<IDType, NNZType, ValueType> *hyperGraph) const = 0;
};

}  // namespace io

}  // namespace sparsebase

#endif  // SPARSEBASE_SPARSEBASE_UTILS_IO_WRITER_H_
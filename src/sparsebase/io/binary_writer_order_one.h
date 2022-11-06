#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"

#include <string>

#ifndef SPARSEBASE_PROJECT_BINARY_WRITER_ORDER_ONE_H
#define SPARSEBASE_PROJECT_BINARY_WRITER_ORDER_ONE_H

namespace sparsebase::io {

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


}
#ifdef _HEADER_ONLY
#include "binary_writer_order_one.cc"
#endif
#endif  // SPARSEBASE_PROJECT_BINARY_WRITER_ORDER_ONE_H

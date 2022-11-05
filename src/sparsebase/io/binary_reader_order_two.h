#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

#include <string>

#ifndef SPARSEBASE_PROJECT_BINARY_READER_ORDER_TWO_H
#define SPARSEBASE_PROJECT_BINARY_READER_ORDER_TWO_H

namespace sparsebase::io {

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
}
#ifdef _HEADER_ONLY
#inclued "binary_reader_order_one.cc"
#endif
#endif  // SPARSEBASE_PROJECT_BINARY_READER_ORDER_TWO_H

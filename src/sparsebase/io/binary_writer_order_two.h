#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"

#include <string>

#ifndef SPARSEBASE_PROJECT_BINARY_WRITER_ORDER_TWO_H
#define SPARSEBASE_PROJECT_BINARY_WRITER_ORDER_TWO_H

namespace sparsebase::io {

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

}
#ifdef _HEADER_ONLY
#inclued "binary_writer_order_two.cc"
#endif
#endif  // SPARSEBASE_PROJECT_BINARY_WRITER_ORDER_TWO_H

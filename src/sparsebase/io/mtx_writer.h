#ifndef SPARSEBASE_PROJECT_MTX_WRITER_H
#define SPARSEBASE_PROJECT_MTX_WRITER_H
#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

//! Writer for the Matrix Market File Format
/*!
 * Detailed explanations of the MTX format can be found in these links:
 * - https://networkrepository.com/mtx-matrix-market-format.html
 * - https://math.nist.gov/MatrixMarket/formats.html
 */
template <typename IDType, typename NNZType, typename ValueType>
class MTXWriter : public Writer,
                             public WritesCOO<IDType, NNZType, ValueType>,
                             public WritesCSR<IDType, NNZType, ValueType>,
                             public WritesArray<ValueType> {
 public:
  explicit MTXWriter(
    std::string filename,
    std::string object = "matrix",
    std::string format = "coordinate",
    std::string field = "real",
    std::string symmetry = "general"
    );
  ~MTXWriter() override = default;
  void WriteCOO(format::COO<IDType, NNZType, ValueType> *coo) const override;
  void WriteCSR(format::CSR<IDType, NNZType, ValueType> *csr) const override;
  void WriteArray(format::Array<ValueType> *arr) const override;
  
 private:
 std::string object_;
  std::string filename_;
  std::string format_;
  std::string field_;
  std::string symmetry_;
};

}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "mtx_writer.cc"
#endif
#endif  // SPARSEBASE_PROJECT_MTX_WRITER_H

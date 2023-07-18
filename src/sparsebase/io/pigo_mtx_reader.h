#ifndef SPARSEBASE_PROJECT_PIGO_MTX_READER_H
#define SPARSEBASE_PROJECT_PIGO_MTX_READER_H

#include <fstream>
#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
namespace sparsebase::io {

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
  // format::Array<ValueType> *ReadArray() const override;
  virtual ~PigoMTXReader() = default;

 private:
  std::string filename_;
  bool weighted_;
  bool convert_to_zero_index_;

  enum MTXObjectOptions { matrix, vector };
  enum MTXFormatOptions { coordinate, array };
  enum MTXFieldOptions { real, double_field, complex, integer, pattern };
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
  MTXOptions options_;
  MTXOptions ParseHeader(std::string header_line) const;
};
}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "pigo_mtx_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PIGO_MTX_READER_H

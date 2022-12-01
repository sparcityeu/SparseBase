#include <fstream>
#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
#ifndef SPARSEBASE_PROJECT_PIGO_EDGE_LIST_READER_H
#define SPARSEBASE_PROJECT_PIGO_EDGE_LIST_READER_H
namespace sparsebase::io {
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
}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "pigo_edge_list_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_PIGO_EDGE_LIST_READER_H

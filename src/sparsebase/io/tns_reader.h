#ifndef SPARSEBASE_PROJECT_TNS_READER_H
#define SPARSEBASE_PROJECT_TNS_READER_H
#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

namespace sparsebase::io {

//! Reader for the TNS format specified in FROSTT.
/*!
 * Detailed explanations of the TNS format can be found in this link:
 * - http://frostt.io/tensors/file-formats.html
 */
template <typename IDType, typename NNZType, typename ValueType>
class TNSReader : public Reader,
                  public ReadsHigherOrderCOO<IDType, NNZType, ValueType>,
                  {
 public:
  /*!
   * Constructor for the TNSReader class
   * @param filename path to the file to be read
   * @param convert_to_zero_index if set to true the indices will be converted
   * such that they start from 0 instead of 1
   */
  explicit TNSReader(std::string filename, bool store_values = true, bool convert_to_zero_index = true);
  format::HigherOrderCOO<IDType, NNZType, ValueType> *ReadHigherOrderCOO() const override;
  ~TNSReader() override;

 private:
  std::string filename_;
  bool convert_to_zero_index_;
  bool store_values_;
};
}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "tns_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_TNS_READER_H

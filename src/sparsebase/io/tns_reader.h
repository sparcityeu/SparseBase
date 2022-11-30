#ifndef SPARSEBASE_PROJECT_TNS_READER_H
#define SPARSEBASE_PROJECT_TNS_READER_H
#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

namespace sparsebase::io {

//! Reader for the TNS format specified in FROSTT.
/*!
 * Detailed explanations of the MTX format can be found in these links:
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
  explicit MTXReader(std::string filename, bool convert_to_zero_index = true);
  format::COO<IDType, NNZType, ValueType> *ReadHigherOrderCOO() const override;
  ~TNSReader() override;

 private:
  format::Array<ValueType> *ReadCoordinateIntoArray() const;
  format::Array<ValueType> *ReadArrayIntoArray() const;
  template <bool weighted>
  format::COO<IDType, NNZType, ValueType> *ReadArrayIntoCOO() const;
  template <bool weighted, int symm, bool conv_to_zero>
  format::COO<IDType, NNZType, ValueType> *ReadCoordinateIntoCOO() const;
  std::string filename_;
  bool convert_to_zero_index_;
};
}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "tns_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_TNS_READER_H

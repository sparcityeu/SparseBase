#ifndef SPARSEBASE_PROJECT_MTX_READER_H
#define SPARSEBASE_PROJECT_MTX_READER_H
#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

namespace sparsebase::io {
#define MMX_PREFIX "%%MatrixMarket"
//! Reader for the Matrix Market File Format
/*!
 * Detailed explanations of the MTX format can be found in these links:
 * - https://networkrepository.com/mtx-matrix-market-format.html
 * - https://math.nist.gov/MatrixMarket/formats.html
 */
template <typename IDType, typename NNZType, typename ValueType>
class MTXReader : public Reader,
                  public ReadsCSR<IDType, NNZType, ValueType>,
                  public ReadsCOO<IDType, NNZType, ValueType>,
                  public ReadsArray<ValueType> {
 public:
  /*!
   * Constructor for the MTXReader class
   * @param filename path to the file to be read
   * @param convert_to_zero_index if set to true the indices will be converted
   * such that they start from 0 instead of 1
   */
  explicit MTXReader(std::string filename, bool convert_to_zero_index = true,
                     bool upper_triangle = false);
  format::COO<IDType, NNZType, ValueType> *ReadCOO() const override;
  format::CSR<IDType, NNZType, ValueType> *ReadCSR() const override;
  format::Array<ValueType> *ReadArray() const override;
  ~MTXReader() override;

 private:
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
  MTXOptions ParseHeader(std::string header_line) const;
  format::Array<ValueType> *ReadCoordinateIntoArray() const;
  format::Array<ValueType> *ReadArrayIntoArray() const;
  template <bool weighted>
  format::COO<IDType, NNZType, ValueType> *ReadArrayIntoCOO() const;
  template <bool weighted, int symm, bool conv_to_zero, bool upper_triangle>
  format::COO<IDType, NNZType, ValueType> *ReadCoordinateIntoCOO() const;
  std::string filename_;
  bool convert_to_zero_index_;
  bool upper_triangle_;
  MTXOptions options_;
};
}  // namespace sparsebase::io
#ifdef _HEADER_ONLY
#include "mtx_reader.cc"
#endif
#endif  // SPARSEBASE_PROJECT_MTX_READER_H

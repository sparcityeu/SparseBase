#include "sparsebase/io/binary_reader_order_two.h"

#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/io/sparse_file_format.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::BinaryReaderOrderTwo(
    std::string filename)
    : filename_(filename) {}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType>
    *BinaryReaderOrderTwo<IDType, NNZType, ValueType>::ReadCSR() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if (sbff.get_name() != "csr") {
    throw utils::ReaderException("SBFF file is not in CSR format");
  }

  NNZType *row_ptr;
  IDType *col;
  ValueType *vals = nullptr;

  auto dimensions = sbff.get_dimensions();

  sbff.template GetArray("row_ptr", row_ptr);
  sbff.template GetArray("col", col);

  if constexpr (!std::is_same_v<ValueType, void>) {
    sbff.template GetArray("vals", vals);
  } else {
    throw utils::ReaderException(
        "Cannot read a weighted COO into a format with void ValueType");
  }

  return new format::CSR<IDType, NNZType, ValueType>(
      dimensions[0], dimensions[1], row_ptr, col, vals, format::kOwned);
}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType>
    *BinaryReaderOrderTwo<IDType, NNZType, ValueType>::ReadCOO() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if (sbff.get_name() != "coo") {
    throw utils::ReaderException("SBFF file is not in COO format");
  }

  IDType *row;
  IDType *col;
  ValueType *vals = nullptr;

  auto dimensions = sbff.get_dimensions();

  sbff.template GetArray("row", row);
  sbff.template GetArray("col", col);

  if (sbff.get_array_count() == 3) {
    if constexpr (!std::is_same_v<ValueType, void>) {
      sbff.template GetArray("vals", vals);
    } else {
      throw utils::ReaderException(
          "Cannot read a weighted COO into a format with void ValueType");
    }
  }

  return new format::COO<IDType, NNZType, ValueType>(
      dimensions[0], dimensions[1], dimensions[1], row, col, vals,
      format::kOwned);
}

#ifndef _HEADER_ONLY
#include "init/binary_reader_order_two.inc"
#endif
}  // namespace sparsebase::io

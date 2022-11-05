#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/io/binary_reader_order_one.h"
#include "sparsebase/io/sparse_file_format.h"

namespace sparsebase::io{
template <typename T>
BinaryReaderOrderOne<T>::BinaryReaderOrderOne(std::string filename)
    : filename_(filename) {
  static_assert(!std::is_same_v<T, void>,
                "A BinaryReaderOrderOne cannot read an Array of type void");
}

template <typename T>
format::Array<T> *BinaryReaderOrderOne<T>::ReadArray() const {
  static_assert(!std::is_same_v<T, void>,
                "A BinaryReaderOrderOne cannot read an Array of type void");
  auto sbff = SbffObject::ReadObject(filename_);

  if (sbff.get_name() != "array") {
    throw utils::ReaderException("SBFF file is not in Array format");
  }

  format::DimensionType size = sbff.get_dimensions()[0];
  T *arr;
  if constexpr (!std::is_same_v<T, void>) {
    sbff.template GetArray("array", arr);
  } else {
    throw utils::ReaderException("Cannot read an into an Array with void ValueType");
  }
  return new format::Array<T>(size, arr, format::kOwned);
}
#ifndef _HEADER_ONLY
#include "init/binary_reader_order_one.inc"
#endif
}

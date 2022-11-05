#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/io/binary_writer_order_one.h"
#include "sparsebase/io/sparse_file_format.h"

namespace sparsebase::io{

template <typename T>
BinaryWriterOrderOne<T>::BinaryWriterOrderOne(std::string filename)
    : filename_(filename) {}

template <typename T>
void BinaryWriterOrderOne<T>::WriteArray(format::Array<T> *arr) const {
  SbffObject sbff("array");
  if constexpr (!std::is_same_v<T, void>) {
    sbff.AddDimensions(arr->get_dimensions());
    sbff.AddArray("array", arr->get_vals(), arr->get_dimensions()[0]);
    sbff.WriteObject(filename_);
  } else {
    throw utils::WriterException("Cannot write an Array with void ValueType");
  }
}
#ifndef _HEADER_ONLY
#include "init/binary_writer_order_one.inc"
#endif
}

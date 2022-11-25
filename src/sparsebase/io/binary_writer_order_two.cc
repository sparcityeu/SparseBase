#include "sparsebase/io/binary_writer_order_two.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
BinaryWriterOrderTwo<IDType, NNZType, ValueType>::BinaryWriterOrderTwo(
    std::string filename)
    : filename_(filename) {}

template <typename IDType, typename NNZType, typename ValueType>
void BinaryWriterOrderTwo<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {
  SbffObject sbff("coo");
  sbff.AddDimensions(coo->get_dimensions());
  sbff.AddArray("row", coo->get_row(), coo->get_num_nnz());
  sbff.AddArray("col", coo->get_col(), coo->get_num_nnz());

  if (coo->get_vals() != nullptr)
    if constexpr (!std::is_same_v<ValueType, void>)
      sbff.AddArray("vals", coo->get_vals(), coo->get_num_nnz());

  sbff.WriteObject(filename_);
}

template <typename IDType, typename NNZType, typename ValueType>
void BinaryWriterOrderTwo<IDType, NNZType, ValueType>::WriteCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) const {
  SbffObject sbff("csr");

  int n, m;
  auto dimensions = csr->get_dimensions();
  n = dimensions[0];
  m = dimensions[1];

  sbff.AddDimensions(dimensions);
  sbff.AddArray("row_ptr", csr->get_row_ptr(), n + 1);
  sbff.AddArray("col", csr->get_col(), m);

  if (csr->get_vals() != nullptr)
    if constexpr (!std::is_same_v<ValueType, void>)
      sbff.AddArray("vals", csr->get_vals(), m);
    else
      throw utils::WriterException("Cannot write vals array of type void");

  sbff.WriteObject(filename_);
}
#ifndef _HEADER_ONLY
#include "init/binary_writer_order_two.inc"
#endif
}  // namespace sparsebase::io

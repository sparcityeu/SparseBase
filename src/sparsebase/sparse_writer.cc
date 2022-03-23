#include "sparsebase/sparse_writer.h"
#include "sparsebase/sparse_exception.h"
#include "sparsebase/sparse_file_format.h"

namespace sparsebase::utils {

template <typename IDType, typename NNZType, typename ValueType>
BinaryWriter<IDType, NNZType, ValueType>::BinaryWriter(std::string filename)
    : filename_(filename) {}


template <typename IDType, typename NNZType, typename ValueType>
void BinaryWriter<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {

  SbffObject sbff("coo");
  sbff.AddArray("row", coo->get_row(), coo->get_num_nnz());
  sbff.AddArray("col", coo->get_col(), coo->get_num_nnz());

  if(coo->get_vals() != nullptr)
    sbff.AddArray("vals", coo->get_vals(), coo->get_num_nnz());

  sbff.WriteObject(filename_);
}

template <typename IDType, typename NNZType, typename ValueType>
void BinaryWriter<IDType, NNZType, ValueType>::WriteCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) const {

  SbffObject sbff("csr");

  int n,m;
  auto dimensions = csr->get_dimensions();
  n = dimensions[0];
  m = dimensions[1];

  sbff.AddArray("row_ptr", csr->get_row_ptr(), n+1);
  sbff.AddArray("col", csr->get_col(), m);

  if(csr->get_vals() != nullptr)
    sbff.AddArray("vals", csr->get_vals(), m);

  sbff.WriteObject(filename_);
}
}

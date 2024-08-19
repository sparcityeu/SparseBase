#include "sparsebase/io/metis_graph_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csr.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
MetisGraphWriter<IDType, NNZType, ValueType>::MetisGraphWriter(
    std::string filename, bool edgeWeighted, bool vertexWeighted,
    bool zero_indexed)
    : filename_(filename),
      edgeWeighted_(edgeWeighted),
      vertexWeighted_(vertexWeighted),
      zero_indexed_(zero_indexed) {}

template <typename IDType, typename NNZType, typename ValueType>
void MetisGraphWriter<IDType, NNZType, ValueType>::WriteGraph(
    object::Graph<IDType, NNZType, ValueType> *graph) const {std::ofstream metisGraphFile;
  metisGraphFile.open(filename_);

  format::Format* con = graph->get_connectivity();
  format::COO<IDType, NNZType, ValueType>* coo = con->AsAbsolute<format::COO<IDType, NNZType, ValueType>>();
  converter::ConverterOrderTwo<IDType, NNZType, ValueType> converterObj;
  context::CPUContext cpu_context;

  format::CSR<IDType, NNZType, ValueType>* csr = converterObj.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, &cpu_context);
  auto n = (csr->get_dimensions()[0] - !zero_indexed_);
  auto m = csr->get_num_nnz() / 2;
  auto row_ptr = csr->get_row_ptr();
  auto col = csr->get_col();
  int NCON = 0;
  std::string FMT = "0";
  metisGraphFile << " " << n << " " << m;
  if constexpr (std::is_same_v<ValueType, void>) {
    metisGraphFile << "\n";

    for (int i = (1 - zero_indexed_); i < csr->get_dimensions()[0]; ++i) {
      for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        metisGraphFile << " " << (col[j] + zero_indexed_) << (j+1 == row_ptr[i+1] ? "" : " ");
      }
      metisGraphFile << "\n";
    }
  }
  else {
    auto val = csr->get_vals();
    auto vertexWeights = graph->vertexWeights_;
    if (vertexWeighted_) NCON = graph->ncon_;
    if (edgeWeighted_ && !vertexWeighted_) FMT = "1";
    else if (edgeWeighted_) FMT = "11";
    else FMT = "10";
    if (NCON > 0 || FMT != "0") {
      metisGraphFile << " " << FMT;
      if (NCON > 0)
        metisGraphFile << " " << NCON;
    }
    metisGraphFile << "\n";

    for (int i = (1 - zero_indexed_); i < csr->get_dimensions()[0]; ++i) {
      if (vertexWeighted_)
      {
        auto weights = vertexWeights[i]->get_vals();
        for (int j = 0; j < NCON; ++j)
          metisGraphFile << weights[j] << " ";
        metisGraphFile << "  ";
      }
      for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
        metisGraphFile << " " << (col[j] + zero_indexed_);
        if (edgeWeighted_)
          metisGraphFile << " " << val[j] << (j+1 == row_ptr[i+1] ? "" : " ");
        if (j+1 != row_ptr[i+1])
          metisGraphFile << " ";
      }
      metisGraphFile << "\n";
    }
  }
  metisGraphFile.close();
}

#ifndef _HEADER_ONLY
#include "init/metis_graph_writer.inc"
#endif
}  // namespace sparsebase::io

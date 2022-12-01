#include "sparsebase/reorder/reorder_heatmap.h"

#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/logger.h"

namespace sparsebase::reorder {
template <typename IDType, typename NNZType, typename ValueType,
          typename FloatType>
ReorderHeatmap<IDType, NNZType, ValueType, FloatType>::ReorderHeatmap() {
  this->params_ = std::make_unique<ReorderHeatmapParams>();
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static(),
       format::Array<IDType>::get_id_static(),
       format::Array<IDType>::get_id_static()},
      ReorderHeatmapCSRArrayArray);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FloatType>
ReorderHeatmap<IDType, NNZType, ValueType, FloatType>::ReorderHeatmap(
    ReorderHeatmapParams params)
    : ReorderHeatmap() {
  this->params_ = std::make_unique<ReorderHeatmapParams>(params);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FloatType>
format::FormatOrderOne<FloatType>*
ReorderHeatmap<IDType, NNZType, ValueType, FloatType>::Get(
    format::FormatOrderTwo<IDType, NNZType, ValueType>* format,
    format::FormatOrderOne<IDType>* permutation_r,
    format::FormatOrderOne<IDType>* permutation_c,
    std::vector<context::Context*> contexts, bool convert_input) {
  return this->Execute(this->params_.get(), contexts, convert_input,
                       (format::Format*)format, (format::Format*)permutation_r,
                       (format::Format*)permutation_c);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FloatType>
format::FormatOrderOne<FloatType>*
ReorderHeatmap<IDType, NNZType, ValueType, FloatType>::
    ReorderHeatmapCSRArrayArray(std::vector<format::Format*> formats,
                                utils::Parameters* poly_params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  auto order_r = formats[1]->AsAbsolute<format::Array<IDType>>()->get_vals();
  auto order_c = formats[2]->AsAbsolute<format::Array<IDType>>()->get_vals();
  auto* params = static_cast<ReorderHeatmapParams*>(poly_params);
  int b = params->num_parts;
  if (b > csr->get_dimensions()[0] || b > csr->get_dimensions()[1]) {
    throw utils::ReorderException(
        "Cannot generate heatmap for matrix when num_parts > number of rows or "
        "columns");
  }
  auto n = csr->get_dimensions()[0];
  auto row_ptr = csr->get_row_ptr();
  auto adj = csr->get_col();
  IDType max_bw = 0;
  FloatType mean_bw = 0;

  IDType bsize = n / b;

  // matrix of size num_parts x num_parts with number of edges in each square
  auto density = new NNZType*[b];
  for (NNZType i = 0; i < b; i++) {
    density[i] = new NNZType[b];
    memset(density[i], 0, sizeof(NNZType) * b);
  }

  for (IDType i = 0; i < n; i++) {
    IDType u = order_r[i];
    IDType bu = u / bsize;
    if (bu >= b) bu = b - 1;
    for (NNZType ptr = row_ptr[i]; ptr < row_ptr[i + 1]; ptr++) {
      IDType v = order_c[adj[ptr]];
      IDType bw =
          abs(std::make_signed_t<IDType>(u) - std::make_signed_t<IDType>(v));
      max_bw = std::max<IDType>(max_bw, bw);
      mean_bw += bw;

      IDType bv = v / bsize;
      if (bv >= b) bv = b - 1;
      density[bu][bv]++;
    }
  }
  mean_bw = (mean_bw + 0.0f) / row_ptr[n];
  // utils::Logger logger;
  // logger.Log("BW stats -- Mean bw: "+std::to_string(mean_bw) + "Max bw: " +
  // std::to_string(max_bw), utils::LOG_LVL_INFO);

  FloatType para_mean_bw = 0;
  mean_bw = 0;
  int fblocks = 0;
  // logger.Log("Printing blocks
  // \n----------------------------------------------" << endl;
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < b; j++) {
      //     cout << std::setprecision(2) << density[i][j] / (row_ptr[n] + .0f)
      //     << "\t";
      if (density[i][j] > 0) {
        fblocks++;
      }
      int bw = std::abs(i - j);
      mean_bw += bw * density[i][j];
    }
    //   cout << endl;
  }
  // cout << "---------------------------------------------------------------"
  // << endl; cout << "Block BW stats -- No full blocks: " << fblocks << " Block
  // BW: " << (mean_bw + 0.0f) / row_ptr[n] << endl; cout <<
  // "---------------------------------------------------------------" << endl;
  auto heat_values = new FloatType[b * b];
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < b; j++) {
      heat_values[i * b + j] = density[i][j] / (row_ptr[n] + .0f);
    }
  }
  return new format::Array<FloatType>(b * b, heat_values, format::kOwned);
}

#if !defined(_HEADER_ONLY)
#include "init/reorder_heatmap.inc"
#endif
}  // namespace sparsebase::reorder

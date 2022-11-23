#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_REORDER_HEATMAP_H
#define SPARSEBASE_PROJECT_REORDER_HEATMAP_H

namespace sparsebase::reorder {

//! Parameters for Reorder Heatmap generator
struct ReorderHeatmapParams : utils::Parameters {
  //! Number of parts to split vertices over
  int num_parts = 3;
  ReorderHeatmapParams(int b) : num_parts(b){}
  ReorderHeatmapParams(){}
};

//! Calculates density of non-zeros of a 2D format on a num_parts * num_parts grid
/*!
 * Splits the input 2D matrix into a grid of size num_parts * num_parts containing an
 * equal number of rows and columns, and calculates the density of non-zeros in each
 * cell in the grid relative to the total number of non-zeros in the matrix, given that the
 * matrix was reordered according to a permutation matrix.
 * Returns the densities as a dense array (FormatOrderOne) of size num_parts * num_parts where
 * the density at cell [i][j] in the 2D grid is located at index [i*num_parts+j] in the
 * grid. The density values sum up to 1.
 * @tparam FloatType type used to represent the densities of non-zeros.
 */
template <typename IDType, typename NNZType, typename ValueType, typename FloatType>
class ReorderHeatmap : public utils::FunctionMatcherMixin<format::FormatOrderOne<FloatType>*>{
public:
ReorderHeatmap();
ReorderHeatmap(ReorderHeatmapParams params);
format::FormatOrderOne<FloatType>* Get(format::FormatOrderTwo<IDType, NNZType, ValueType> *format, format::FormatOrderOne<IDType>* permutation_r, format::FormatOrderOne<IDType>* permutation_c, std::vector<context::Context*> contexts, bool convert_input);
protected:
static format::FormatOrderOne<FloatType>* ReorderHeatmapCSRArrayArray(std::vector<format::Format*> formats, utils::Parameters * poly_params);
};


}
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/reorder_heatmap.cc"
#endif
#endif  // SPARSEBASE_PROJECT_REORDER_HEATMAP_H

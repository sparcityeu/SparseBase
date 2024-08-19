#include <utility>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_PROJECT_GRAY_REORDER_H
#define SPARSEBASE_PROJECT_GRAY_REORDER_H

namespace sparsebase::reorder {

enum BitMapSize{
  BitSize16 = 16,
  BitSize32 = 32,
  BitSize64 = 64
};
//! Params struct for GrayReorder
struct GrayReorderParams : utils::Parameters {
  BitMapSize resolution;
  int nnz_threshold;
  int sparse_density_group_size;
  explicit GrayReorderParams() {}
  GrayReorderParams(BitMapSize r, int nnz_thresh, int group_size)
      : resolution(r),
        nnz_threshold(nnz_thresh),
        sparse_density_group_size(group_size) {}
};

template <typename IDType, typename NNZType, typename ValueType>
class GrayReorder : public Reorderer<IDType> {
  typedef std::pair<IDType, unsigned long> row_grey_pair;

 public:
  //! Parameter type for GrayReorder
  typedef GrayReorderParams ParamsType;
  GrayReorder(BitMapSize resolution, int nnz_threshold,
              int sparse_density_group_size);
  explicit GrayReorder(GrayReorderParams);

 protected:
  static bool desc_comparator(const row_grey_pair &l, const row_grey_pair &r);

  static bool asc_comparator(const row_grey_pair &l, const row_grey_pair &r);

  // not sure if all IDTypes work for this
  static unsigned long long grey_bin_to_dec(unsigned long long n);

  static void print_dec_in_bin(unsigned long n, int size);

  // not sure if all IDTypes work for this
  static unsigned long bin_to_grey(unsigned long n);
  static bool is_banded(int nnz, int n_cols, NNZType *row_ptr, IDType *cols,
                        std::vector<IDType> order, int band_size = -1);

  static IDType *GrayReorderingCSR(std::vector<format::Format *> input_sf,
                                   utils::Parameters *poly_params);
};

}  // namespace sparsebase::reorder
#ifdef _HEADER_ONLY
#include "sparsebase/reorder/gray_reorder.cc"
#endif
#endif  // SPARSEBASE_PROJECT_GRAY_REORDER_H

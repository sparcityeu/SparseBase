#include "sparsebase/permute/permute_order_one.h"

#include "sparsebase/format/array.h"

namespace sparsebase::permute {
template <typename IDType, typename ValueType>
PermuteOrderOne<IDType, ValueType>::PermuteOrderOne(ParamsType params) {
  PermuteOrderOne(params.order);
}
template <typename IDType, typename ValueType>
PermuteOrderOne<IDType, ValueType>::PermuteOrderOne(IDType *order) {
  this->RegisterFunction({format::Array<ValueType>::get_id_static()},
                         PermuteArray);
  this->params_ = std::make_unique<PermuteOrderOneParams<IDType>>(order);
}
template <typename IDType, typename ValueType>
format::FormatOrderOne<ValueType>
    *PermuteOrderOne<IDType, ValueType>::PermuteArray(
        std::vector<format::Format *> formats, utils::Parameters *params) {
  auto *sp = formats[0]->AsAbsolute<format::Array<ValueType>>();
  auto order = static_cast<PermuteOrderOneParams<IDType> *>(params)->order;
  std::vector<format::DimensionType> dimensions = sp->get_dimensions();
  IDType length = dimensions[0];
  ValueType *vals = sp->get_vals();
  ValueType *nvals = new ValueType[length]();
  IDType *inv_order = new IDType[length];
  for (IDType i = 0; i < length; i++) {
    inv_order[order[i]] = i;
  }

  for (IDType i = 0; i < length; i++) {
    nvals[i] = vals[inv_order[i]];
  }
  format::Array<ValueType> *arr =
      new format::Array<ValueType>(length, nvals, format::kOwned);
  return arr;
}
#if !defined(_HEADER_ONLY)
#include "init/permute_order_one.inc"
#endif
}  // namespace sparsebase::permute

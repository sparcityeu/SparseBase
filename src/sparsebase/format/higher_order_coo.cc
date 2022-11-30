
#include "sparsebase/format/higher_order_coo.h"

#include "sparsebase/utils/logger.h"
namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
HigherOrderCOO<IDType, NNZType, ValueType>::HigherOrderCOO(
    DimensionType order, DimensionType *dimensions, NNZType nnz,
    IDType **indices, ValueType *vals, Ownership own, bool ignore_sort)
    : indices_(indices, BlankDeleter<IDType *>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->nnz_ = nnz;
  this->order_ = order;
  this->dimension_.insert(this->dimension_.begin(), dimensions,
                          dimensions + order);

  if (own == kOwned) {
    this->indices_ = std::unique_ptr<IDType *, std::function<void(IDType **)>>(
        indices, [&](IDType **indices) {
          for (int i = 0; i < this->dimension_.size(); i++)
            delete[] this->indices_.get()[i];
          delete[] this->indices_.get();
        });
    this->vals_ =
        std::unique_ptr<ValueType[], std::function<void(ValueType *)>>(
            vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);

  bool not_sorted = false;
  if (!ignore_sort) {
    for (DimensionType i = 1; i < nnz; i++) {
      for (DimensionType j = 0; j < order; j++) {
        if (indices[j][i - 1] != indices[j][i]) {
          not_sorted = indices[j][i - 1] > indices[j][i];
          break;
        }
      }

      if (not_sorted) break;
    }
  }

  if (not_sorted) {
    std::cerr << "HigherOrderCOO arrays must be sorted. Sorting..."
              << std::endl;
    std::vector<std::tuple<std::vector<IDType>, ValueType>> sort_vec;

    for (DimensionType i = 0; i < nnz; i++) {
      ValueType value = (vals != nullptr) ? vals[i] : 0;

      std::vector<IDType> c_indices;
      for (DimensionType j = 0; j < order; j++)
        c_indices.push_back(indices[j][i]);

      sort_vec.emplace_back(c_indices, value);
    }

    std::sort(sort_vec.begin(), sort_vec.end(),
              [order](std::tuple<std::vector<IDType>, ValueType> t1,
                      std::tuple<std::vector<IDType>, ValueType> t2) {
                for (DimensionType j = 0; j < order; j++) {
                  if (std::get<0>(t1)[j] != std::get<0>(t2)[j])
                    return std::get<0>(t1)[j] < std::get<0>(t2)[j];
                }
                return false;
              });

    for (DimensionType i = 0; i < nnz; i++) {
      auto &t = sort_vec[i];
      for (DimensionType j = 0; j < order; j++) {
        indices[j][i] = std::get<0>(t)[j];
      }

      if (vals != nullptr) {
        vals[i] = std::get<1>(t);
      }
    }
  }
}
template <typename IDType, typename NNZType, typename ValueType>
Format *HigherOrderCOO<IDType, NNZType, ValueType>::Clone() const {
  return nullptr;
  // return new HigherOrderCOO(*this);
}

template <typename IDType, typename NNZType, typename ValueType>
HigherOrderCOO<IDType, NNZType, ValueType>::~HigherOrderCOO(){};

template <typename IDType, typename NNZType, typename ValueType>
IDType **HigherOrderCOO<IDType, NNZType, ValueType>::get_indices() const {
  return indices_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *HigherOrderCOO<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}

#ifndef _HEADER_ONLY
#include "init/higher_order_coo.inc"
#endif
}  // namespace sparsebase::format

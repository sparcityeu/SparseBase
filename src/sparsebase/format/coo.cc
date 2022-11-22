#include "sparsebase/format/coo.h"
#include "sparsebase/utils/logger.h"
namespace sparsebase::format {
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(COO<IDType, NNZType, ValueType> &&rhs)
    : col_(std::move(rhs.col_)),
      row_(std::move(rhs.row_)),
      vals_(std::move(rhs.vals_)) {
  this->nnz_ = rhs.get_num_nnz();
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  rhs.col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      nullptr, BlankDeleter<IDType>());
  rhs.vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      nullptr, BlankDeleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType> &COO<IDType, NNZType, ValueType>::operator=(
    const COO<IDType, NNZType, ValueType> &rhs) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row() + rhs.get_num_nnz(), row);
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<ValueType, void>) {
    if (rhs.get_vals() != nullptr) {
      vals = new ValueType[rhs.get_num_nnz()];
      std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
    }
  }
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  return *this;
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(const COO<IDType, NNZType, ValueType> &rhs)
    : col_(nullptr, BlankDeleter<IDType>()),
      row_(nullptr, BlankDeleter<IDType>()),
      vals_(nullptr, BlankDeleter<ValueType>()) {
  this->nnz_ = rhs.nnz_;
  this->order_ = 2;
  this->dimension_ = rhs.dimension_;
  auto col = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_col(), rhs.get_col() + rhs.get_num_nnz(), col);
  auto row = new IDType[rhs.get_num_nnz()];
  std::copy(rhs.get_row(), rhs.get_row() + rhs.get_num_nnz(), row);
  ValueType *vals = nullptr;
  if constexpr (!std::is_same_v<ValueType, void>) {
    if (rhs.get_vals() != nullptr) {
      vals = new ValueType[rhs.get_num_nnz()];
      std::copy(rhs.get_vals(), rhs.get_vals() + rhs.get_num_nnz(), vals);
    }
  }
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, Deleter<IDType>());
  this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      row, Deleter<IDType>());
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, Deleter<ValueType>());
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);
}
template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::COO(IDType n, IDType m, NNZType nnz,
                                     IDType *row, IDType *col, ValueType *vals,
                                     Ownership own, bool ignore_sort)
    : col_(col, BlankDeleter<IDType>()),
      row_(row, BlankDeleter<IDType>()),
      vals_(vals, BlankDeleter<ValueType>()) {
  this->nnz_ = nnz;
  this->order_ = 2;
  this->dimension_ = {(DimensionType)n, (DimensionType)m};
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, Deleter<IDType>());
    this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        row, Deleter<IDType>());
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  }
  this->context_ = std::unique_ptr<sparsebase::context::Context>(
      new sparsebase::context::CPUContext);

  bool not_sorted = false;
  if (!ignore_sort) {
    IDType prev_row = 0;
    IDType prev_col = 0;
    for (DimensionType i = 0; i < nnz; i++) {
      if (prev_row > row[i] || (prev_row == row[i] && prev_col > col[i])) {
        not_sorted = true;
        break;
      }
      prev_row = row[i];
      prev_col = col[i];
    }
  }

  if (not_sorted) {
    utils::Logger logger(typeid(this));
    logger.Log("COO arrays must be sorted. Sorting...", utils::LOG_LVL_WARNING);

    if constexpr (std::is_same_v<ValueType, void>) {
      std::vector<std::pair<IDType, IDType>> sort_vec;
      for (DimensionType i = 0; i < nnz; i++) {
        sort_vec.emplace_back(row[i], col[i]);
      }
      std::sort(sort_vec.begin(), sort_vec.end(),
                [](std::pair<IDType, IDType> t1, std::pair<IDType, IDType> t2) {
                  if (t1.first == t2.first) {
                    return t1.second < t2.second;
                  }
                  return t1.first < t2.first;
                });

      for (DimensionType i = 0; i < nnz; i++) {
        auto &t = sort_vec[i];
        row[i] = t.first;
        col[i] = t.second;
      }
    } else {
      std::vector<std::tuple<IDType, IDType, ValueType>> sort_vec;
      for (DimensionType i = 0; i < nnz; i++) {
        ValueType value = (vals != nullptr) ? vals[i] : 0;
        sort_vec.emplace_back(row[i], col[i], value);
      }
      std::sort(sort_vec.begin(), sort_vec.end(),
                [](std::tuple<IDType, IDType, ValueType> t1,
                   std::tuple<IDType, IDType, ValueType> t2) {
                  if (std::get<0>(t1) == std::get<0>(t2)) {
                    return std::get<1>(t1) < std::get<1>(t2);
                  }
                  return std::get<0>(t1) < std::get<0>(t2);
                });

      for (DimensionType i = 0; i < nnz; i++) {
        auto &t = sort_vec[i];
        row[i] = std::get<0>(t);
        col[i] = std::get<1>(t);

        if (vals != nullptr) {
          vals[i] = std::get<2>(t);
        }
      }
    }
  }
}
template <typename IDType, typename NNZType, typename ValueType>
Format *COO<IDType, NNZType, ValueType>::Clone() const {
  return new COO(*this);
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::get_col() const {
  return col_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::get_row() const {
  return row_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *COO<IDType, NNZType, ValueType>::get_vals() const {
  return vals_.get();
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::release_col() {
  auto col = col_.release();
  this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      col, BlankDeleter<IDType>());
  return col;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *COO<IDType, NNZType, ValueType>::release_row() {
  auto row = row_.release();
  this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
      row, BlankDeleter<IDType>());
  return row;
}
template <typename IDType, typename NNZType, typename ValueType>
ValueType *COO<IDType, NNZType, ValueType>::release_vals() {
  auto vals = vals_.release();
  this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
      vals, BlankDeleter<ValueType>());
  return vals;
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_col(IDType *col, Ownership own) {
  if (own == kOwned) {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, Deleter<IDType>());
  } else {
    this->col_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        col, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_row(IDType *row, Ownership own) {
  if (own == kOwned) {
    this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        row, Deleter<IDType>());
  } else {
    this->row_ = std::unique_ptr<IDType, std::function<void(IDType *)>>(
        row, BlankDeleter<IDType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
void COO<IDType, NNZType, ValueType>::set_vals(ValueType *vals, Ownership own) {
  if (own == kOwned) {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, Deleter<ValueType>());
  } else {
    this->vals_ = std::unique_ptr<ValueType, std::function<void(ValueType *)>>(
        vals, BlankDeleter<ValueType>());
  }
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::RowIsOwned() {
  return (this->row_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::ColIsOwned() {
  return (this->col_.get_deleter().target_type() !=
          typeid(BlankDeleter<IDType>));
}

template <typename IDType, typename NNZType, typename ValueType>
bool COO<IDType, NNZType, ValueType>::ValsIsOwned() {
  return (this->vals_.get_deleter().target_type() !=
          typeid(BlankDeleter<ValueType>));
}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType>::~COO(){};

#ifndef _HEADER_ONLY
#include "init/coo.inc"
#endif
}

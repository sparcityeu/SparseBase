#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"

#include "sparsebase/format/format_order_one.h"
#ifndef SPARSEBASE_PROJECT_ARRAY_H
#define SPARSEBASE_PROJECT_ARRAY_H
namespace sparsebase::format {
//! One dimensional Format class that wraps a native C++ array
/*!
 * This class basically functions as a wrapper for native C++ arrays such that
 * they can be used polymorphically with the rest of the Format classes
 * @tparam ValueType type that the array stores
 */
template <typename ValueType>
class Array
    : public utils::IdentifiableImplementation<Array<ValueType>,
                                               FormatOrderOne<ValueType>> {
 public:
  Array(DimensionType nnz, ValueType *row_ptr, Ownership own = kNotOwned);
  Array(const Array<ValueType> &);
  Array(Array<ValueType> &&);
  Array<ValueType> &operator=(const Array<ValueType> &);
  Format *Clone() const override;
  virtual ~Array();
  ValueType *get_vals() const;

  ValueType *release_vals();

  void set_vals(ValueType *, Ownership own = kNotOwned);

  virtual bool ValsIsOwned();

 protected:
  std::unique_ptr<ValueType, std::function<void(ValueType *)>> vals_;
};
template <typename ValueType>
template <typename ToValueType>
struct FormatOrderOne<ValueType>::TypeConverter<format::Array, ToValueType> {
  format::Array<ToValueType> *operator()(FormatOrderOne<ValueType> *source,
                                         bool is_move_conversion) {
    auto arr = source->template As<format::Array>();
    auto num_nnz = arr->get_num_nnz();
    ToValueType *new_vals;
    if (!is_move_conversion || !std::is_same_v<ValueType, ToValueType>) {
      new_vals = utils::ConvertArrayType<ToValueType>(arr->get_vals(), num_nnz);
    } else {
      if constexpr (std::is_same_v<ValueType, ToValueType>) {
        new_vals = arr->release_vals();
      }
    }
    return new Array<ToValueType>(num_nnz, new_vals, kOwned);
  }
};
}

#ifdef _HEADER_ONLY
#include "array.cc"
#endif
#endif  // SPARSEBASE_PROJECT_ARRAY_H

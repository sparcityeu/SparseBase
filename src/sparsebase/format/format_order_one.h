
#include "sparsebase/format/format_implementation.h"
#include <cxxabi.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"
#include "sparsebase/converter/converter_order_two.h"

#ifndef SPARSEBASE_PROJECT_FORMAT_ORDER_ONE_H
#define SPARSEBASE_PROJECT_FORMAT_ORDER_ONE_H
namespace sparsebase::format {
template <typename ValueType>
class FormatOrderOne
    : public FormatImplementation {
 public:
  FormatOrderOne();


  //! Converts `this` to a FormatOrderOne object of type ToType<ValueType>
  /*!
   * @param to_context context used to carry out the conversion.
   * @param is_move_conversion whether to carry out a move conversion.
   * @return If `this` is of type `ToType<ValueType>` then it returns the same
   * object but as a different type. If not, it will convert `this` to a new
   * FormatOrderOne object and return a pointer to the new object.
   */
  template <template <typename> typename ToType>
  ToType<ValueType> *Convert(context::Context *to_context = nullptr,
                             bool is_move_conversion = false);

  template <template <typename> typename ToType>
  ToType<ValueType> *Convert(const std::vector<context::Context *> &to_context,
                             bool is_move_conversion = false);

  template <template <typename> typename ToType, typename ToValueType>
  ToType<ToValueType> *Convert(bool is_move_conversion = false);

  template <template <typename> typename ToType, typename ToValueType>
  struct TypeConverter {
    ToType<ToValueType> *operator()(FormatOrderOne<ValueType> *, bool) {
      static_assert(utils::always_false<ToValueType>,
                    "Cannot do type conversion for the requested type. Throw a "
                    "rock through one of our devs' windows");
    }
  };
  template <typename T>
  typename std::remove_pointer<T>::type *As() {
    static_assert(utils::always_false<T>,
                  "When casting a FormatOrderTwo, only pass the class name "
                  "without its types");
    return nullptr;
  }
  template <template <typename> typename T>
  typename std::remove_pointer<T<ValueType>>::type *As() {
    static_assert(std::is_base_of_v<FormatOrderOne<ValueType>, T<ValueType>>,
    "Cannot cast to a non-FormatOrderOne class");
    using TBase = typename std::remove_pointer<T<ValueType>>::type;
    if (this->get_id() == std::type_index(typeid(TBase))) {
      return static_cast<TBase *>(this);
    }
    throw utils::TypeException(this->get_name(),
                               typeid(TBase).name());
  }
};
template <typename ValueType>
template <template <typename> class ToType>
ToType<ValueType> *sparsebase::format::FormatOrderOne<ValueType>::Convert(
    context::Context *to_context, bool is_move_conversion) {
  static_assert(std::is_base_of<format::FormatOrderOne<ValueType>,
      ToType<ValueType>>::value,
      "T must be a format::Format");
  //auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  context::Context *actual_context =
      to_context == nullptr ? this->get_context() : to_context;
  return converter
      ->Convert(this, ToType<ValueType>::get_id_static(), actual_context,
                is_move_conversion)
      ->template AsAbsolute<ToType<ValueType>>();
}
template <typename ValueType>
template <template <typename> class ToType>
ToType<ValueType> *sparsebase::format::FormatOrderOne<ValueType>::Convert(
    const std::vector<context::Context *> &to_contexts,
    bool is_move_conversion) {
  static_assert(std::is_base_of<format::FormatOrderOne<ValueType>,
      ToType<ValueType>>::value,
      "T must be a format::Format");
  //auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  std::vector<context::Context *> vec = {this->get_context()};
  std::vector<context::Context *> actual_contexts =
      to_contexts.empty() ? vec : to_contexts;
  return converter
      ->Convert(this, ToType<ValueType>::get_id_static(), actual_contexts,
                is_move_conversion)
      ->template AsAbsolute<ToType<ValueType>>();
}
template <typename ValueType>
template <template <typename> typename ToType, typename ToValueType>
ToType<ToValueType> *FormatOrderOne<ValueType>::Convert(
    bool is_move_conversion) {
  static_assert(std::is_base_of<format::FormatOrderOne<ToValueType>,
      ToType<ToValueType>>::value,
      "T must be an order one format");

  //auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  if (this->get_id() != ToType<ValueType>::get_id_static()) {
    auto converted_format = converter->template Convert<ToType<ValueType>>(
        this, this->get_context(), is_move_conversion);
    auto type_converted_format = TypeConverter<ToType, ToValueType>()(
        converted_format, is_move_conversion);
    delete converted_format;
    return type_converted_format;
  } else {
    return TypeConverter<ToType, ToValueType>()(this, is_move_conversion);
  }
}
}
#ifdef _HEADER_ONLY
#include "format_order_one.cc"
#endif
#endif  // SPARSEBASE_PROJECT_FORMAT_ORDER_ONE_H

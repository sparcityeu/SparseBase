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
#include "sparsebase/converter/converter_order_two.h"
#include "sparsebase/format/format_implementation.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"
#ifndef SPARSEBASE_PROJECT_FORMAT_ORDER_TWO_H
#define SPARSEBASE_PROJECT_FORMAT_ORDER_TWO_H
namespace sparsebase::format {

template <typename IDType, typename NNZType, typename ValueType>
class FormatOrderTwo : public FormatImplementation {
 public:
  FormatOrderTwo();

  //! Converts `this` to a FormatOrderTwo object of type ToType<IDType, NNZType,
  //! ValueType>
  /*!
   * @param to_context context used to carry out the conversion.
   * @param is_move_conversion whether to carry out a move conversion.
   * @return If `this` is of type `ToType<ValueType>` then it returns the same
   * object but as a different type. If not, it will convert `this` to a new
   * FormatOrderOne object and return a pointer to the new object.
   */
  template <template <typename, typename, typename> class ToType>
  ToType<IDType, NNZType, ValueType> *Convert(
      context::Context *to_context = nullptr, bool is_move_conversion = false);

  template <template <typename, typename, typename> class ToType>
  ToType<IDType, NNZType, ValueType> *Convert(
      const std::vector<context::Context *> &to_context,
      bool is_move_conversion = false);

  template <template <typename, typename, typename> class ToType,
            typename ToIDType, typename ToNNZType, typename ToValueType>
  ToType<ToIDType, ToNNZType, ToValueType> *Convert(
      bool is_move_conversion = false);
  template <template <typename, typename, typename> class ToType,
            typename ToIDType, typename ToNNZType, typename ToValueType>
  struct TypeConverter {
    ToType<ToIDType, ToNNZType, ToValueType> *operator()(
        FormatOrderTwo<IDType, NNZType, ValueType> *, bool) {
      static_assert(utils::always_false<ToIDType>,
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
  template <template <typename, typename, typename> typename T>
  typename std::remove_pointer<T<IDType, NNZType, ValueType>>::type *As() {
    static_assert(std::is_base_of_v<FormatOrderTwo<IDType, NNZType, ValueType>,
                                    T<IDType, NNZType, ValueType>>,
                  "Cannot cast to a non-FormatOrderTwo class");
    using TBase =
        typename std::remove_pointer<T<IDType, NNZType, ValueType>>::type;
    if (this->get_id() == std::type_index(typeid(TBase))) {
      return static_cast<TBase *>(this);
    }
    throw utils::TypeException(this->get_name(), typeid(TBase).name());
  }
  template <template <typename, typename, typename> typename T>
  bool Is() {
    using TBase = typename std::remove_pointer<T<IDType, NNZType, ValueType>>::type;
    return this->get_id() == std::type_index(typeid(TBase));
	}
};
template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType>
ToType<IDType, NNZType, ValueType>
    *FormatOrderTwo<IDType, NNZType, ValueType>::Convert(
        context::Context *to_context, bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<IDType, NNZType, ValueType>,
                      ToType<IDType, NNZType, ValueType>>::value,
      "T must be an order two format");
  // converter::Converter* converter = this->converter_.get();
  auto converter = this->get_converter();
  context::Context *actual_context =
      to_context == nullptr ? this->get_context() : to_context;
  return converter
      ->Convert(this, ToType<IDType, NNZType, ValueType>::get_id_static(),
                actual_context, is_move_conversion)
      ->template AsAbsolute<ToType<IDType, NNZType, ValueType>>();
}

template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType>
ToType<IDType, NNZType, ValueType>
    *FormatOrderTwo<IDType, NNZType, ValueType>::Convert(
        const std::vector<context::Context *> &to_contexts,
        bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<IDType, NNZType, ValueType>,
                      ToType<IDType, NNZType, ValueType>>::value,
      "T must be an order two format");
  // auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  std::vector<context::Context *> vec = {this->get_context()};
  std::vector<context::Context *> actual_contexts =
      to_contexts.empty() ? vec : to_contexts;
  return converter
      ->Convert(this, ToType<IDType, NNZType, ValueType>::get_id_static(),
                actual_contexts, is_move_conversion)
      ->template AsAbsolute<ToType<IDType, NNZType, ValueType>>();
}

template <typename IDType, typename NNZType, typename ValueType>
template <template <typename, typename, typename> class ToType,
          typename ToIDType, typename ToNNZType, typename ToValueType>
ToType<ToIDType, ToNNZType, ToValueType> *
FormatOrderTwo<IDType, NNZType, ValueType>::Convert(bool is_move_conversion) {
  static_assert(
      std::is_base_of<format::FormatOrderTwo<ToIDType, ToNNZType, ToValueType>,
                      ToType<ToIDType, ToNNZType, ToValueType>>::value,
      "T must be an order two format");
  // auto* converter = this->converter_.get();
  auto converter = this->get_converter();
  if (this->get_id() != ToType<IDType, NNZType, ValueType>::get_id_static()) {
    auto converted_format =
        converter->template Convert<ToType<IDType, NNZType, ValueType>>(
            this, this->get_context(), is_move_conversion);
    auto type_converted_format =
        TypeConverter<ToType, ToIDType, ToNNZType, ToValueType>()(
            converted_format, is_move_conversion);
    delete converted_format;
    return type_converted_format;
  } else {
    return TypeConverter<ToType, ToIDType, ToNNZType, ToValueType>()(
        this, is_move_conversion);
  }
}
}  // namespace sparsebase::format
#ifdef _HEADER_ONLY
#include "format_order_two.cc"
#endif
#endif  // SPARSEBASE_PROJECT_FORMAT_ORDER_TWO_H

/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_UTILS_H_
#define SPARSEBASE_SPARSEBASE_UTILS_H_

#include <any>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "exception.h"

namespace sparsebase::utils {

namespace MatrixMarket {

enum MTXObjectOptions { matrix, vector };
enum MTXFormatOptions { coordinate, array };
enum MTXFieldOptions { real, double_field, complex, integer, pattern };
enum MTXSymmetryOptions {
  general = 0,
  symmetric = 1,
  skew_symmetric = 2,
  hermitian = 3
};
struct MTXOptions {
  MTXObjectOptions object;
  MTXFormatOptions format;
  MTXFieldOptions field;
  MTXSymmetryOptions symmetry;
};
MTXOptions ParseHeader(std::string header_line);

}  // namespace MatrixMarket

//! Type used for calculating function costs
typedef float CostType;
// Thanks to artificial mind blog:
// https://artificial-mind.net/blog/2020/10/03/always-false
template <typename... T>
constexpr bool always_false = false;

//! Functor used for hashing vectors of type_index values.
struct TypeIndexVectorHash {
  std::size_t operator()(const std::vector<std::type_index> &vf) const;
};

using std::numeric_limits;

// Cross float-integral type conversion is not currently available
template <typename T, typename U>
bool CanTypeFitValue(const U value) {
  bool decision;
  if constexpr (std::is_integral_v<T> != std::is_integral_v<U>)
    decision = false;
  if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {
    const intmax_t botT = []() {
      intmax_t ret;
      if constexpr (std::is_floating_point_v<T>)
        ret = intmax_t(-(numeric_limits<T>::max()));
      else
        ret = intmax_t(numeric_limits<T>::min());
      return ret;
    }();
    const intmax_t botU = []() {
      intmax_t ret;
      if constexpr (std::is_floating_point_v<U>)
        ret = intmax_t(-(numeric_limits<U>::max()));
      else
        ret = intmax_t(numeric_limits<U>::min());
      return ret;
    }();
    const uintmax_t topT = uintmax_t(numeric_limits<T>::max());
    const uintmax_t topU = uintmax_t(numeric_limits<U>::max());
    decision = !((botT > botU && value < (U)(botT)) ||
                 (topT < topU && value > (U)(topT)));
  } else if constexpr (!std::is_integral_v<T> && !std::is_integral_v<U>) {
    const double botT = []() {
      T ret;
      if constexpr (std::is_floating_point_v<T>)
        ret = T(-(numeric_limits<T>::max()));
      else
        ret = T(numeric_limits<T>::min());
      return ret;
    }();
    const double botU = []() {
      U ret;
      if constexpr (std::is_floating_point_v<U>)
        ret = U(-(numeric_limits<U>::max()));
      else
        ret = U(numeric_limits<U>::min());
      return ret;
    }();
    const double topT = numeric_limits<T>::max();
    const double topU = numeric_limits<U>::max();
    decision = !((botT > botU && value < (U)(botT)) ||
                 (topT < topU && value > (U)(topT)));
  }
  return decision;
  //} else if constexpr (std::is_integral_v<T> && !std::is_integral_v<U> ){
  //  const double topT = double(numeric_limits<T>::max());
  //  const uintmax_t topU = uintmax_t(numeric_limits<U>::max());
  //  const double botT = []() {
  //    if constexpr (std::is_floating_point_v<T>)
  //      return double(-(numeric_limits<T>::max()));
  //    else
  //      return double(numeric_limits<T>::min());
  //  }();
  //  const intmax_t botU = []() {
  //    if constexpr (std::is_floating_point_v<U>)
  //      return intmax_t(-(numeric_limits<U>::max()));
  //    else
  //      return intmax_t(numeric_limits<U>::min());
  //  }();
  //  return !(double(topU) > topT && double(value) > topT) || !(double(botU) <
  //  botT && double(value) > topT);
  //} else {
  //  const uintmax_t topT = uintmax_t(numeric_limits<T>::max());
  //  const double topU = double(numeric_limits<U>::max());
  //  const intmax_t botT = []() {
  //    if constexpr (std::is_floating_point_v<T>)
  //      return intmax_t(-(numeric_limits<T>::max()));
  //    else
  //      return intmax_t(numeric_limits<T>::min());
  //  }();
  //  const double botU = []() {
  //    if constexpr (std::is_floating_point_v<U>)
  //      return double(-(numeric_limits<U>::max()));
  //    else
  //      return double(numeric_limits<U>::min());
  //  }();
  //  return !(topU > double(topT) && value > double(topT)) || !(botU <
  //  double(botT) && value > double(topT));
  //}
}

template <typename FromType, typename ToType>
inline bool isTypeConversionSafe(FromType from_val, ToType to_val) {
  return from_val == to_val && CanTypeFitValue<ToType>(from_val);
}

template <typename ToType, typename FromType, typename SizeType>
ToType *ConvertArrayType(FromType *from_ptr, SizeType size) {
  if constexpr (!(std::is_same_v<ToType, void> &&
                  std::is_same_v<FromType, void>)) {
    if (from_ptr == nullptr) return nullptr;
    auto to_ptr = new ToType[size];
    for (SizeType i = 0; i < size; i++) {
      to_ptr[i] = from_ptr[i];
      if (!isTypeConversionSafe(from_ptr[i], to_ptr[i])) {
        throw utils::TypeException(
            "Could not convert array from type " +
            std::string(std::type_index(typeid(FromType)).name()) +
            " to type " + std::string(std::type_index(typeid(ToType)).name()) +
            ". Overflow detected");
      }
    }
    return to_ptr;
  }
  return nullptr;
}

template <typename T>
class OnceSettable {
 public:
  OnceSettable() : is_set_(false) {}
  operator T() const { return data_; }
  OnceSettable(const OnceSettable &) = delete;
  OnceSettable(OnceSettable &&) = delete;
  OnceSettable &operator=(T &&data) {
    if (!is_set_) {
      data_ = std::move(data);
      is_set_ = true;
      return *this;
    }
    throw utils::AttemptToReset<T>();
  }
  const T &get() const { return data_; }

 private:
  T data_;
  bool is_set_;
};
std::string demangle(const std::string &name);

std::string demangle(std::type_index type);

class Identifiable {
 public:
  //! Returns the std::type_index for the concrete Format class that this
  //! instance is a member of
  virtual std::type_index get_id() const = 0;

  virtual std::string get_name() const = 0;
};
template <typename IdentifiableType, typename Base>
class IdentifiableImplementation : public Base {
 public:
  //! Returns the std::type_index for the concrete Format class that this
  //! instance is a member of
  virtual std::type_index get_id() const { return typeid(IdentifiableType); }

  virtual std::string get_name() const { return utils::demangle(get_id()); };

  //! A static variant of the get_id() function
  static std::type_index get_id_static() { return typeid(IdentifiableType); }

  static std::string get_name_static() {
    return utils::demangle(get_id_static());
  };
};

template <typename Interface>
struct Implementation {
 public:
  Implementation() = default;
  template <typename ConcreteType>
  explicit Implementation(ConcreteType &&object)
      : storage{std::forward<ConcreteType>(object)},
        getter{[](std::any &storage) -> Interface & {
          return std::any_cast<ConcreteType &>(storage);
        }} {}
  Implementation(const Implementation &object)
      : storage{object.storage}, getter{object.getter} {}
  Implementation(Implementation &&object) noexcept
      : storage{std::move(object.storage)}, getter{std::move(object.getter)} {}
  Implementation &operator=(Implementation other) {
    storage = other.storage;
    getter = other.getter;
    return *this;
  }

  Interface *operator->() { return &getter(storage); }

 private:
  std::any storage;
  Interface &(*getter)(std::any &);
};

}  // namespace sparsebase::utils

#ifdef _HEADER_ONLY
#include "sparsebase/utils/utils.cc"
#endif

#endif
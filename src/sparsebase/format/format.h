/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_FORMAT_FORMAT_H_
#define SPARSEBASE_SPARSEBASE_FORMAT_FORMAT_H_

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
#include "sparsebase/context/context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"

namespace sparsebase {

namespace converter {
class Converter;
template <typename ValueType>
class ConverterOrderOne;
template <typename IDType, typename NNZType, typename ValueType>
class ConverterOrderTwo;
}

namespace format {

//! Enum depicting the ownership status of a Format instance
enum Ownership {
  //! When used the arrays are owned by the user (Format instance is not
  //! responsible from deallocation)
  kNotOwned = 0,
  //! When used the arrays are owned and will be deallocated by the Format
  //! instance
  kOwned = 1,
};

template <typename T>
struct Deleter {
  void operator()(T *obj) {
    if constexpr (!std::is_same_v<T, void>)
      if (obj != nullptr) delete obj;
  }
};

template <long long dim, typename T>
struct Deleter2D {
  void operator()(T *obj) {
    for (long long i = 0; i < dim; i++) {
      delete[] obj[i];
    }
    delete[] obj;
  }
};

template <class T>
struct BlankDeleter {
  void operator()(T *obj) {}
};

//! Type used to represent sizes in the library (usually will be a 64-bit
//! unsigned integer type)
typedef unsigned long long DimensionType;

//! Base class representing a Sparse Data Format (like CSR, COO etc.)
/*!
 * Format is base class representing a Sparse Data Format.
 * All Sparse Data Format Objects (like CSR and COO) are derived from Format.
 * Most of the library uses Format pointers as input and output.
 * By itself Format has very little functionality besides giving size
 * information. Often a user will have to use the AsAbsolute() function to
 * convert it to a concrete format.
 */
class Format : public utils::Identifiable {
 public:
  //! Returns a type identifier for the concrete format class
  /*!
   * @return A type_index object from the standard C++ library that represents
   * the type of the underlying concrete object. For example, a CSR and a COO
   * object will have different identifiers. But also, objects of the same class
   * with varying template types (like CSR<int,int,int> and CSR<int,int,float)
   * will also have different identifiers.
   */
  //virtual std::type_index get_id() = 0;

  ////! Similar to get_id but returns a demangled name instead
  //virtual std::string get_name() = 0;

  virtual ~Format() = default;

  //! Performs a deep copy of the Format object and returns the pointer to the
  //! newly created object
  virtual Format *Clone() const = 0;

  //! Returns the sizes of all dimensions as a vector
  virtual std::vector<DimensionType> get_dimensions() const = 0;

  //! Returns the number of non-zero elements stored
  virtual DimensionType get_num_nnz() const = 0;

  //! Returns the number of dimensions
  virtual DimensionType get_order() const = 0;

  //! Returns the context for the format instance
  virtual context::Context *get_context() const = 0;

  //! Returns the type_index for the context for the format instance
  virtual std::type_index get_id() const = 0;

  //! Returns a pointer at the converter of this format instance
  virtual std::shared_ptr<converter::Converter const> get_converter() const = 0;
  
  //! Sets a new converter for this format instance
  //virtual void set_converter(converter::Converter*) = 0;

  //! Templated function that can be used to cast to a concrete format class
  /*!
   * @tparam T a concrete format class (for example: CSR<int,int,int>)
   * @return A concrete format pointer to this object (for example:
   * CSR<int,int,int>*)
   */
  template <typename T>
  typename std::remove_pointer<T>::type *AsAbsolute() {
    static_assert(std::is_base_of_v<Format, T>,
                  "Cannot cast a non-Format class using AsAbsolute");
    using TBase = typename std::remove_pointer<T>::type;
    if (this->get_id() == std::type_index(typeid(TBase))) {
      return static_cast<TBase *>(this);
    }
    throw utils::TypeException(get_name(), typeid(TBase).name());
  }
  template <template <typename...> typename T>
  void *AsAbsolute() {
    static_assert(utils::always_false<T<void>>,
                  "When casting a format pointer, you need to pass the data "
                  "type with the template arguments");
    return nullptr;
  }

  //! Templated function that can be used to check the concrete type of this
  //! object
  /*!
   * @tparam T a concrete format class (for example: CSR<int,int,int>)
   * @return true if the type of this object is T
   */
  template <typename T>
  bool Is() {
    using TBase = typename std::remove_pointer<T>::type;
    return this->get_id() == std::type_index(typeid(TBase));
  }
};



}  // namespace format

}  // namespace sparsebase

#include "sparsebase/converter/converter.h"
#include "sparsebase/converter/converter_order_one.h"
#include "sparsebase/converter/converter_order_two.h"

namespace sparsebase::format {



}  // namespace sparsebase
//#include "sparsebase/converter/converter.h"

//namespace sparsebase::format {
//template <typename FormatType, typename Base>
//void FormatImplementation<FormatType, Base>::set_converter(converter::Converter* ptr) {
//  this->converter_ = std::shared_ptr<converter::Converter>(ptr->Clone());
//}
//}
#endif

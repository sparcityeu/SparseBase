/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#ifndef SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_
#define SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_

#include <any>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/class_matcher_mixin.h"
#include "sparsebase/utils/utils.h"

namespace sparsebase::feature {


using Feature = utils::Implementation<preprocess::ExtractableType>;


//! Extractor provides an interface for users to generate multiple features
//! optimally with a single call.
/*!
 *  Detailed
 */
class Extractor : public utils::ClassMatcherMixin<preprocess::ExtractableType *> {
 public:
  ~Extractor();
  //! Computes the features that are passed.
  /*!
    Detailed Description.
    @param features vector of features to extract.
    @param format a format to be used as the data source.
    @param con vector of contexts to be used to determine the where the
    computation will take place. @return void
  */
  static std::unordered_map<std::type_index, std::any> Extract(
      std::vector<Feature> &features, format::Format *format,
      const std::vector<context::Context *> &, bool convert_input);
  std::
      unordered_map<std::type_index, std::any>
      //! Computes the features that are added to in_ private data member.
      /*!
        Detailed Description.
        @param format a format to be used as the data source.
        @param con vector of contexts to be used to determine the where the
        computation will take place. @return void
      */
      Extract(format::Format *format,
              const std::vector<context::Context *> &con, bool convert_input);
  //! Adds a feature to private in_ data member.
  /*!
    Detailed Description.
    @param f a Feature argument.
    @return void
  */
  void Add(Feature f);
  //! Subtracts a feature from private in_ data member.
  /*!
    Detailed Description.
    @param f a Feature argument.
    @return void
  */
  void Subtract(Feature f);
  //! Returns the in_ private data member as a vector.
  /*!
    Detailed Description.
    @return vector of type std::type_index
  */
  std::vector<std::type_index> GetList();
  //! Prints all the registered functions to the ClassMatcher map.
  /*!
    Detailed Description.
    @return void
  */
  void PrintFuncList();
  std::vector<preprocess::ExtractableType *> GetFuncList();

 protected:
  Extractor() noexcept = default;

 private:
  //! Stores the features that are going to be extracted once the Extract
  //! function is called.
  /*!
   *  Detailed
   */
  std::unordered_map<std::type_index, preprocess::ExtractableType *> in_;
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
class FeatureExtractor : public Extractor {
 public:
  FeatureExtractor();
};

}  // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparsebase/feature/feature.cc"
#endif

#endif  // SPARSEBASE_SPARSEBASE_FEATURE_FEATURE_H_
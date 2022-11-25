/*******************************************************
 * Copyright (c) 2022 SparCity, Amro Alabsi Aljundi, Taha Atahan Akyildiz, Arda
 *Sener All rights reserved.
 *
 * This file is distributed under MIT license.
 * The complete license agreement can be obtained at:
 * https://sparcityeu.github.io/sparsebase/pages/license.html
 ********************************************************/
#include <sparsebase/context/context.h>

#include <memory>

#include "sparsebase/utils/parameterizable.h"
#ifndef SPARSEBASE_SPARSEBASE_UTILS_EXTRACTABLE_H_
#define SPARSEBASE_SPARSEBASE_UTILS_EXTRACTABLE_H_

namespace sparsebase::utils {

//! Abstract class that can be utilized with fusued feature extraction
/*!
 * Classes implementing Extractable can be used with with a
 * sparsebase::feature::Extractor for fused feature extraction. Each
 * Extractable object can be a fusion of multiple Extractable classes.
 * An Extractable object will contain parameters for each of the
 * Extractable it is fusud into as well as one for itself.
 */
class Extractable {
 public:
  //! Extract features from the passed Format through passed Contexts
  /*!
   *
   * \param format object from which features are extracted.
   * \param contexts vector of contexts that can be used for extracting
   * features. \return An uordered map containing the extracted features as
   * key-value pairs with the key being the std::type_index of the feature and
   * the value an std::any to that feature.
   */
  virtual std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *> contexts,
      bool convert_input) = 0;
  //! Returns the std::type_index of this class
  virtual std::type_index get_id() = 0;
  //! Get the std::type_index of all the Extractable classes fused into this
  //! class
  /*!
   *
   * \return a vector containing the std::type_index values of all the
   * Extractable classes fusued into this class
   */
  virtual std::vector<std::type_index> get_sub_ids() = 0;
  //! Get instances of the Extractable classes that make up this class
  /*!
   * \return A vector of pointers to Extractable objects, each of which
   * corresponds to one of the features that this class is extracting, and the
   * classes will have their respective parameters passed over to them.
   */
  virtual std::vector<Extractable *> get_subs() = 0;
  //! Get a std::shared_ptr at the Parameters of this object
  /*!
   *
   * \return An std::shared_ptr at the same Parameters instance of this
   * object (not a copy)
   */
  virtual std::shared_ptr<utils::Parameters> get_params() = 0;
  //! Get an std::shared_ptr at a Parameters of one of the Extractable
  //! classes fused into this class
  /*!
   * Returns a std::shared_ptr at a Parameters object belonging to one of
   * the Extractable classes fused into this class \param feature_extractor
   * std::type_index identifying the Extractable within this class whose
   * parameters are requested \return an std::shared_ptr at the Parameters
   * corresponding feature_extractor
   */
  virtual std::shared_ptr<utils::Parameters> get_params(
      std::type_index feature_extractor) = 0;
  //! Set the parameters of one of Extractable classes fusued into this
  //! classes.
  /*!
   * \param feature_extractor std::type_index identifying the Extractable
   * class fusued into this class whose parameters are to be set. \param params
   * an std::shared_ptr at the Parameters belonging to the class
   * feature_extractor
   */
  virtual void set_params(std::type_index feature_extractor,
                          std::shared_ptr<utils::Parameters> params) = 0;
  virtual ~Extractable() = default;

 protected:
  //! a pointer at the Parameters of this class
  std::shared_ptr<utils::Parameters> params_;
  //! A key-value map of Parameters, one for each of the Extractable
  //! classes fused into this class
  std::unordered_map<std::type_index, std::shared_ptr<utils::Parameters>> pmap_;
};

}  // namespace sparsebase::utils
#endif
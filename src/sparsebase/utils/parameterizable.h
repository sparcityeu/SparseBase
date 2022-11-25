//
// Created by Amro on 11/5/2022.
//

#ifndef SPARSEBASE_PROJECT_PARAMETERIZABLE_H
#define SPARSEBASE_PROJECT_PARAMETERIZABLE_H
#include <memory>

namespace sparsebase::utils {
//! An abstraction for parameter objects used for preprocessing
struct Parameters {};

//! A generic type for all preprocessing types
class Parameterizable {
  //! The parameter class used to pass parameters to this preprocessing
  typedef Parameters ParamsType;

 protected:
  //! Polymorphic pointer at a PreprocessParams object
  std::unique_ptr<Parameters> params_;
};
}  // namespace sparsebase::utils

#endif  // SPARSEBASE_PROJECT_PARAMETERIZABLE_H

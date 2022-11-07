#include "sparsebase/format/format.h"
#ifndef SPARSEBASE_PROJECT_FORMAT_IMPLEMENTATION_H
#define SPARSEBASE_PROJECT_FORMAT_IMPLEMENTATION_H
namespace sparsebase::format{

//! A class derived from the base Format class, mostly used for development
//! purposes
/*!
 * FormatImplementation derives from the Format class.
 * It implements some common functionality used by all formats.
 * The Curiously recurring template pattern (CRTP) is used here for retrieving
 * type information of the subclasses (like CSR and COO).
 *
 *
 * Under normal circumstances users should not interact with this class directly
 * unless they are defining their own format in which case this class should be
 * derived from by passing the newly defined format as a template parameter.
 *
 * @tparam FormatType used for CRTP, should be a concrete format class
 * (for example: CSR<int,int,int>)
 */
class FormatImplementation : public Format {
 public:
  virtual std::vector<DimensionType> get_dimensions() const {
    return this->dimension_;
  }
  virtual DimensionType get_num_nnz() const { return this->nnz_; }
  virtual DimensionType get_order() const { return this->order_; }
  virtual context::Context *get_context() const { return this->context_.get().get(); }

  virtual std::type_index get_id() const {
    return this->context_.get()->get_id();
  }
  virtual std::shared_ptr<converter::Converter const> get_converter()
  const {
    std::cout << "getting converter " << this->converter_.get() << std::endl;
    return this->converter_;
    //return std::dynamic_pointer_cast<converter::Converter>(this->converter_).get();
  };
  void set_converter(std::shared_ptr<converter::Converter> converter)
  {
    this->converter_ = converter;
  };
 protected:
  DimensionType order_;
  std::vector<DimensionType> dimension_;
  DimensionType nnz_;
  utils::OnceSettable<std::unique_ptr<sparsebase::context::Context>> context_;
  std::shared_ptr<converter::Converter> converter_;
};
}
#endif  // SPARSEBASE_PROJECT_FORMAT_IMPLEMENTATION_H

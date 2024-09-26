#include "sparsebase/config.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/utils.h"
#ifndef SPARSEBASE_PROJECT_HIGHER_ORDER_COO_H
#define SPARSEBASE_PROJECT_HIGHER_ORDER_COO_H
namespace sparsebase::format {

//! Higher Order Coordinate List Sparse Data Format
/*!
 * @tparam IDType type used for the dimensions
 * @tparam NNZType type used for non-zeros and the number of non-zeros
 * @tparam ValueType type used for the stored values
 */
template <typename IDType, typename NNZType, typename ValueType>
class HigherOrderCOO : public utils::IdentifiableImplementation<
                           HigherOrderCOO<IDType, NNZType, ValueType>,
                           FormatImplementation> {
 public:
  HigherOrderCOO(DimensionType order, DimensionType *dimensions, NNZType nnz,
                 IDType **indices, ValueType *vals, Ownership own = kNotOwned,
                 bool ignore_sort = false);

  // HigherOrderCOO(const HigherOrderCOO<IDType, NNZType, ValueType> &);
  // HigherOrderCOO(HigherOrderCOO<IDType, NNZType, ValueType> &&);

  // HigherOrderCOO<IDType, NNZType, ValueType> & operator=(const
  // HigherOrderCOO<IDType, NNZType, ValueType> &);

  Format *Clone() const override;
  virtual ~HigherOrderCOO();

  IDType **get_indices() const;
  ValueType *get_vals() const;

  // IDType **release_indices();
  // ValueType *release_vals();

  // void set_indices(IDType **, Ownership own = kNotOwned);
  // void set_vals(ValueType *, Ownership own = kNotOwned);

  // virtual bool IndicesIsOwned();
  // virtual bool ValsIsOwned();

 protected:
  std::unique_ptr<IDType *, std::function<void(IDType **)>> indices_;
  std::unique_ptr<ValueType[], std::function<void(ValueType *)>> vals_;
};

}  // namespace sparsebase::format

#ifdef _HEADER_ONLY
#include "higher_order_coo.cc"
#endif
#endif  // SPARSEBASE_PROJECT_HigherOrderCOO_H

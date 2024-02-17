#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_BANDWIDTH_H
#define SPARSEBASE_PROJECT_BANDWIDTH_H
namespace sparsebase::feature {
//! Bandwidth calculation class for graph representation
template <typename IDType, typename NNZType, typename ValueType>
class Bandwidth : public feature::FeaturePreprocessType<int *> {
 public:
  //! An empty struct used for the parameters of Bandwidth
  typedef utils::Parameters ParamsType;
  Bandwidth();
  Bandwidth(ParamsType);
  Bandwidth(const Bandwidth<IDType, NNZType, ValueType> &d);
  Bandwidth(std::shared_ptr<ParamsType>);
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<utils::Extractable *> get_subs() override;
  static std::type_index get_id_static();

    //! Calculate the bandwidth of the format
    /*!
    * @param format a single format pointer to any format
    * @param contexts vector of contexts that can be used for extracting
    * features.
    * @param convert_input whether or not to convert the input format if that
    * is needed.
    * \return calculated bandwidth of graph representation of `format`
    */
  int *GetBandwidth(format::Format *format,
                     std::vector<context::Context *> contexts,
                     bool convert_input);
  std::
      tuple<std::vector<std::vector<format::Format *>>, int *>
      //! Calculate the bandwidth of the format with cached output
      /*!
       *
       * @param format a single format pointer to any format
       * @param contexts vector of contexts that can be used for extracting
       * features.
       * @param convert_input whether or not to convert the input format if that
       * is needed.
       * @return calculated bandwidth of graph representation of `format`
       */
      GetBandwidthCached(format::Format *format,
                       std::vector<context::Context *> contexts,
                       bool convert_input);
  //! Bandwidth implementation function for CSRs
  /*!
   *
   * @param formats A vector containing a single format pointer that should
   * point at a CSR object
   * @param params a Parameters pointer, though it
   * is not used in the function
   * @return calculated bandwidth of graph representation of `format[0]`
   */
  static int *GetBandwidthCSR(std::vector<format::Format *> formats,
                               utils::Parameters *params);
  ~Bandwidth();

 protected:
  void Register();
};

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/bandwidth.cc"
#endif
#endif  // SPARSEBASE_PROJECT_BANDWIDTH_H
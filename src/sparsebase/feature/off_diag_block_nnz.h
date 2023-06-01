#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"

#ifndef SPARSEBASE_PROJECT_OFF_DIAG_BLOCK_NNZ_H
#define SPARSEBASE_PROJECT_OFF_DIAG_BLOCK_NNZ_H
namespace sparsebase::feature {

struct OffDiagBlockNNZParams : utils::Parameters {
  int blockrowsize = 1, blockcolsize = 1;
  OffDiagBlockNNZParams() {}
  OffDiagBlockNNZParams(int N) : blockrowsize(N), blockcolsize(N){}
  OffDiagBlockNNZParams(int blockrowsize, int blockcolsize)
      : blockrowsize(blockrowsize), blockcolsize(blockcolsize) {}
};

//! Find the number of nonzeros in off-diagonal blocks of partitioned matrix, in the graph representation of a format object
template <typename IDType, typename NNZType, typename ValueType>
class OffDiagBlockNNZ : public feature::FeaturePreprocessType<IDType *> {
 public:
  typedef OffDiagBlockNNZParams ParamsType;
  OffDiagBlockNNZ();
  OffDiagBlockNNZ(ParamsType);
  OffDiagBlockNNZ(const OffDiagBlockNNZ<IDType, NNZType, ValueType> &d);
  OffDiagBlockNNZ(std::shared_ptr<ParamsType>);
  std::unordered_map<std::type_index, std::any> Extract(
      format::Format *format, std::vector<context::Context *>,
      bool convert_input) override;
  std::vector<std::type_index> get_sub_ids() override;
  std::vector<utils::Extractable *> get_subs() override;
  static std::type_index get_id_static();

  //! OffDiagBlockNNZ executor function that carries out function matching
  /*!
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting features.
   * @param convert_input whether or not to convert the input format if that is needed.
   * @return OffDiagBlockNNZ in the graph representation of `format`
   */
  IDType *GetOffDiagBlockNNZ(format::Format *format, std::vector<context::Context *> contexts,
                     bool convert_input);

  std::tuple<std::vector<std::vector<format::Format *>>, IDType *>
  //! OffDiagBlockNNZ executor function that carries out function matching with cached output
  /*!
   * @param format a single format pointer to any format
   * @param contexts vector of contexts that can be used for extracting features.
   * @param convert_input whether or not to convert the input format if that is needed.
   * @return OffDiagBlockNNZ in the graph representation of `format`
   */
  GetOffDiagBlockNNZCached(format::Format *format, std::vector<context::Context *> contexts,
                       bool convert_input);
  //! OffDiagBlockNNZ implementation function for CSRs
  /*!
   * @param formats A vector containing a single format pointer that should point at a CSR object
   * @param params a Parameters pointer, though it is not used in the function
   * @return OffDiagBlockNNZ in the graph representations of 'format[0]'
   */
  static IDType *GetOffDiagBlockNNZCSR(std::vector<format::Format *> formats, utils::Parameters *params);

  ~OffDiagBlockNNZ();

 protected:
  void Register();
};

}  // namespace sparsebase::feature
#ifdef _HEADER_ONLY
#include "sparsebase/feature/off_diag_block_nnz.cc"
#endif
#endif  // SPARSEBASE_PROJECT_OFF_DIAG_BLOCK_NNZ_H

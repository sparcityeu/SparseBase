#include <any>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/class_matcher_mixin.h"
#include "sparsebase/utils/extractable.h"
#include "sparsebase/utils/utils.h"
#ifndef SPARSEBASE_PROJECT_EXTRACTOR_H
#define SPARSEBASE_PROJECT_EXTRACTOR_H
namespace sparsebase::feature {
using Feature = utils::Implementation<utils::Extractable>;

//! Extractor provides an interface for users to generate multiple features
//! optimally with a single call.
/*!
 *  Detailed
 */
class Extractor : public utils::ClassMatcherMixin<utils::Extractable *> {
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
  std::vector<utils::Extractable *> GetFuncList();

 protected:
  Extractor() noexcept = default;

 private:
  //! Stores the features that are going to be extracted once the Extract
  //! function is called.
  /*!
   *  Detailed
   */
  std::unordered_map<std::type_index, utils::Extractable *> in_;
};
}  // namespace sparsebase::feature

#ifdef _HEADER_ONLY
#include "sparsebase/feature/extractor.cc"
#endif
#endif  // SPARSEBASE_PROJECT_EXTRACTOR_H

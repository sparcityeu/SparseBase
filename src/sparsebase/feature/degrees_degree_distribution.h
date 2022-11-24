#include <vector>

#include "sparsebase/config.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/utils/parameterizable.h"
#include "sparsebase/feature/feature_preprocess_type.h"

#ifndef SPARSEBASE_PROJECT_DEGREES_DEGREE_DISTRIBUTION_H
#define SPARSEBASE_PROJECT_DEGREES_DEGREE_DISTRIBUTION_H
namespace sparsebase::feature {
//! An empty struct used for the parameters of Degrees_DegreeDistribution
struct Params : utils::Parameters {};
//! Find the degree and degree distribution of each vertex in the graph
//! representation of a format object
template <typename IDType, typename NNZType, typename ValueType,
    typename FeatureType>
class Degrees_DegreeDistribution
    : public feature::FeaturePreprocessType<
        std::unordered_map<std::type_index, std::any>> {
//! An empty struct used for the parameters of Degrees_DegreeDistribution
typedef Params ParamsType;

public:
Degrees_DegreeDistribution();
Degrees_DegreeDistribution(Params);
Degrees_DegreeDistribution(
    const Degrees_DegreeDistribution<IDType, NNZType, ValueType, FeatureType>
    &d);
Degrees_DegreeDistribution(std::shared_ptr<Params>);
std::unordered_map<std::type_index, std::any> Extract(
    format::Format *format, std::vector<context::Context *>,
    bool convert_input) override;
std::vector<std::type_index> get_sub_ids() override;
std::vector<utils::Extractable *> get_subs() override;
static std::type_index get_id_static();

//! Degree and degree distribution generation executor function that carries
//! out function matching
/*!
 *
 * @param format a single format pointer to any format
 * @param contexts vector of contexts that can be used for extracting
 * features.
 * @param convert_input whether or not to convert the input format if that is
 * needed.
 * @return a map with two (type_index, any) pairs. One is a degrees
 * array of type IDType*, and one is a degree distribution array of type
 * FeatureType*. Both arrays have the respective metric of the ith vertex in
 * the ith array element.
 */
std::unordered_map<std::type_index, std::any> Get(
    format::Format *format, std::vector<context::Context *> contexts,
    bool convert_input);

//! Degree and degree distribution implementation function for CSRs
/*!
 *
 * @param format a single format pointer to any format
 * @param params a utils::Parameters pointer, though it
 * is not used in the function
 * features. @return a map with two (type_index, any) pairs. One is a degrees
 * array of type IDType*, and one is a degree distribution array of type
 * FeatureType*. Both arrays have the respective metric of the ith vertex in
 * the ith array element.
 */
static std::unordered_map<std::type_index, std::any> GetCSR(
    std::vector<format::Format *> formats, utils::Parameters *params);
~Degrees_DegreeDistribution();

protected:
void Register();
};

}
#ifdef _HEADER_ONLY
#include "sparsebase/feature/degrees_degree_distribution.cc"
#endif

#endif  // SPARSEBASE_PROJECT_DEGREES_DEGREE_DISTRIBUTION_H

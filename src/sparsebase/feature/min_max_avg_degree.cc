#include <algorithm>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/feature/min_max_avg_degree.h"
#include "sparsebase/feature/min_degree.h"
#include "sparsebase/feature/max_degree.h"
#include "sparsebase/feature/avg_degree.h"
#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MinMaxAvgDegree<IDType, NNZType, ValueType,
                           FeatureType>::MinMaxAvgDegree(Params) {
  MinMaxAvgDegree();
}
template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MinMaxAvgDegree<IDType, NNZType, ValueType,
                           FeatureType>::MinMaxAvgDegree() {
  this->Register();
  this->params_ = std::shared_ptr<Params>(new Params());
  this->pmap_.insert({get_id_static(), this->params_});


  std::shared_ptr<utils::Parameters> min_deg_param(new utils::Parameters);
  this->pmap_[MinDegree<IDType, NNZType, ValueType>::get_id_static()] =
      min_deg_param;

  std::shared_ptr<utils::Parameters> max_deg_param(new utils::Parameters);
  this->pmap_[MinDegree<IDType, NNZType, ValueType>::get_id_static()] =
      max_deg_param;

  std::shared_ptr<utils::Parameters> avg_deg_param(
      new AvgDegreeParams);
  this->pmap_[AvgDegree<IDType, NNZType, ValueType,
                                 FeatureType>::get_id_static()] =
      avg_deg_param;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
void MinMaxAvgDegree<IDType, NNZType, ValueType,
                                FeatureType>::Register() {
  this->RegisterFunction(
      std::vector<std::type_index>(
          {format::CSR<IDType, NNZType, ValueType>::get_id_static()}),
      GetCSR);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>::
    MinMaxAvgDegree(const MinMaxAvgDegree<
                               IDType, NNZType, ValueType, FeatureType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>::
    MinMaxAvgDegree(const std::shared_ptr<Params> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
MinMaxAvgDegree<IDType, NNZType, ValueType,
                           FeatureType>::~MinMaxAvgDegree() =
    default;

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<std::type_index> MinMaxAvgDegree<
    IDType, NNZType, ValueType, FeatureType>::get_sub_ids() {
  std::vector<std::type_index> r = {
      typeid(MinDegree<IDType, NNZType, ValueType>),
      typeid(MaxDegree<IDType, NNZType, ValueType>),
      typeid(AvgDegree<IDType, NNZType, ValueType, FeatureType>)};
  std::sort(r.begin(), r.end());
  return r;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::vector<utils::Extractable *> MinMaxAvgDegree<
    IDType, NNZType, ValueType, FeatureType>::get_subs() {
  auto *f1 = new MinDegree<IDType, NNZType, ValueType>();
  if (this->pmap_.find(MinDegree<IDType, NNZType, ValueType>::get_id_static()) !=
      this->pmap_.end()) {
    f1->set_params(
        MinDegree<IDType, NNZType, ValueType>::get_id_static(),
        this->pmap_[MinDegree<IDType, NNZType, ValueType>::get_id_static()]);
  }
  auto *f2 = new MaxDegree<IDType, NNZType, ValueType>();
  if (this->pmap_.find(MinDegree<IDType, NNZType, ValueType>::get_id_static()) !=
      this->pmap_.end()) {
    f2->set_params(
        MaxDegree<IDType, NNZType, ValueType>::get_id_static(),
        this->pmap_[MinDegree<IDType, NNZType, ValueType>::get_id_static()]);
  }

  auto *f3 = new AvgDegree<IDType, NNZType, ValueType, FeatureType>();
  if (this->pmap_.find(AvgDegree<IDType, NNZType, ValueType,
                                          FeatureType>::get_id_static()) !=
      this->pmap_.end()) {
    f3->set_params(
        AvgDegree<IDType, NNZType, ValueType,
                           FeatureType>::get_id_static(),
        this->pmap_[AvgDegree<IDType, NNZType, ValueType,
                                       FeatureType>::get_id_static()]);
  }

  std::vector<utils::Extractable *> res(3);
  auto ids = this->get_sub_ids();
  for (int i = 0; i < 3; ++i) {
    if (ids[i] == f1->get_id()) {
      res[i] = f1;
    }
    else if (ids[i] == f2->get_id()) {
      res[i] = f2;
    }
    else {
      res[i] = f3;
    }
  }
  return res;
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::type_index MinMaxAvgDegree<IDType, NNZType, ValueType,
                                           FeatureType>::get_id_static() {
  return typeid(
      MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return Get(format, c, convert_input);
};

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>::Get(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  Params params;
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
std::unordered_map<std::type_index, std::any>
MinMaxAvgDegree<IDType, NNZType, ValueType, FeatureType>::GetCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();
  IDType num_vertices = csr->get_dimensions()[0];
  auto *rows = csr->get_row_ptr();
  NNZType *min_degree = new NNZType();
  *min_degree = (rows[1] - rows[0]);
  NNZType *max_degree = new NNZType();
  *max_degree = (rows[1] - rows[0]);
  FeatureType *avg_degree = new FeatureType();
  for (int i = 1; i < num_vertices; i++) {
    *min_degree = std::min(*min_degree, rows[i + 1] - rows[i]);
    *max_degree = std::max(*max_degree, rows[i + 1] - rows[i]);
  }
  NNZType degree_sum = rows[num_vertices] - rows[0];
  *avg_degree = degree_sum / (FeatureType) num_vertices;
  return {{MinDegree<IDType, NNZType, ValueType>::get_id_static(),
           std::forward<NNZType *>(min_degree)},
          {MaxDegree<IDType, NNZType, ValueType>::get_id_static(),
           std::forward<NNZType *>(max_degree)},
          {AvgDegree<IDType, NNZType, ValueType,
                              FeatureType>::get_id_static(),
           std::forward<FeatureType *>(avg_degree)}};
}

#if !defined(_HEADER_ONLY)
#include "init/min_max_avg_degree.inc"
#endif
}  // namespace sparsebase::feature

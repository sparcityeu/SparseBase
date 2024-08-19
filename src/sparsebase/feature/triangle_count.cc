#include "sparsebase/feature/triangle_count.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsebase/utils/parameterizable.h"

namespace sparsebase::feature {

template <typename IDType, typename NNZType, typename ValueType>
TriangleCount<IDType, NNZType, ValueType>::TriangleCount(ParamsType p) {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType(p.countDirected));
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
TriangleCount<IDType, NNZType, ValueType>::TriangleCount() {
  Register();
  this->params_ = std::shared_ptr<ParamsType>(new ParamsType());
  this->pmap_.insert({get_id_static(), this->params_});
}

template <typename IDType, typename NNZType, typename ValueType>
TriangleCount<IDType, NNZType, ValueType>::TriangleCount(
    const TriangleCount<IDType, NNZType, ValueType> &d) {
  Register();
  this->params_ = d.params_;
  this->pmap_ = d.pmap_;
}

template <typename IDType, typename NNZType, typename ValueType>
TriangleCount<IDType, NNZType, ValueType>::TriangleCount(
    const std::shared_ptr<ParamsType> r) {
  Register();
  this->params_ = r;
  this->pmap_[get_id_static()] = r;
}

template <typename IDType, typename NNZType, typename ValueType>
TriangleCount<IDType, NNZType, ValueType>::~TriangleCount() = default;

template <typename IDType, typename NNZType, typename ValueType>
void TriangleCount<IDType, NNZType, ValueType>::Register() {
  this->RegisterFunction(
      {format::CSR<IDType, NNZType, ValueType>::get_id_static()},
      GetTriangleCountCSR);
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<std::type_index>
TriangleCount<IDType, NNZType, ValueType>::get_sub_ids() {
  return {typeid(TriangleCount<IDType, NNZType, ValueType>)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::vector<utils::Extractable *>
TriangleCount<IDType, NNZType, ValueType>::get_subs() {
  return {new TriangleCount<IDType, NNZType, ValueType>(*this)};
}

template <typename IDType, typename NNZType, typename ValueType>
std::type_index TriangleCount<IDType, NNZType, ValueType>::get_id_static() {
  return typeid(TriangleCount<IDType, NNZType, ValueType>);
}

template <typename IDType, typename NNZType, typename ValueType>
std::unordered_map<std::type_index, std::any>
TriangleCount<IDType, NNZType, ValueType>::Extract(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return {{this->get_id(), std::forward<int64_t *>(
                               GetTriangleCount(format, c, convert_input))}};
};

template <typename IDType, typename NNZType, typename ValueType>
int64_t *TriangleCount<IDType, NNZType, ValueType>::GetTriangleCount(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->Execute(this->params_.get(), c, convert_input, format);
}

template <typename IDType, typename NNZType, typename ValueType>
std::tuple<std::vector<std::vector<format::Format *>>, int64_t *>
TriangleCount<IDType, NNZType, ValueType>::GetTriangleCountCached(
    format::Format *format, std::vector<context::Context *> c,
    bool convert_input) {
  return this->CachedExecute(this->params_.get(), c, convert_input, false,
                             format);
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *TransposeCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) {
  IDType n1 = csr->get_dimensions()[0], n2 = csr->get_dimensions()[1];
  NNZType m = csr->get_num_nnz();
  IDType *indegree = new IDType[n2];
  for (int i = 0; i < n2; ++i) indegree[i] = 0;
  NNZType *row_ptr = csr->get_row_ptr();
  IDType *col = csr->get_col();
  ValueType *val = csr->get_vals();
  for (int i = 0; i < n1; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      ++indegree[col[j]];
    }
  }

  int t_n1 = n2, t_n2 = n1;
  NNZType *t_row_ptr = new NNZType[t_n1 + 1];
  t_row_ptr[0] = 0;
  for (int i = 0; i < t_n1; ++i) {
    t_row_ptr[i + 1] = indegree[i] + t_row_ptr[i];
    indegree[i] = 0;
  }

  IDType *t_col = new IDType[m];
  ValueType *t_val = nullptr;

  if constexpr (!std::is_same_v<ValueType, void>) {
    t_val = new ValueType[m];
  }
  for (int i = 0; i < n1; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      t_col[t_row_ptr[col[j]] + indegree[col[j]]++] = i;
      if constexpr (!std::is_same_v<ValueType, void>) {
        t_val[t_row_ptr[col[j]] + indegree[col[j]] - 1] = val[j];
      }
    }
  }

  delete[] indegree;
  return new format::CSR<IDType, NNZType, ValueType>(
      t_n1, m, t_row_ptr, t_col, t_val, format::kOwned, true);
}

template <typename IDType, typename NNZType, typename ValueType>
int64_t DirectedTriangleCount(format::CSR<IDType, NNZType, ValueType> *csr) {
  int n = csr->get_dimensions()[0];
  NNZType *row_ptr = csr->get_row_ptr();
  IDType *col = csr->get_col();

  auto t_csr = TransposeCSR(csr);
  auto t_row_ptr = t_csr->get_row_ptr();
  auto t_col = t_csr->get_col();

  int *incoming = new int[n];
  for (int i = 0; i < n; ++i) {
    incoming[i] = 0;
  }

  int64_t triangleCount = 0;
  for (int node = 0; node < n; ++node) {
    for (int i = t_row_ptr[node]; i < t_row_ptr[node + 1]; ++i) {
      incoming[t_col[i]] = node;  // t_col[i] -> node
    }

    for (int i = row_ptr[node]; i < row_ptr[node + 1]; ++i) {
      int neig = col[i];
      if (node < neig) {
        for (int j = row_ptr[neig]; j < row_ptr[neig + 1]; ++j) {
          if (node < col[j] && incoming[col[j]])  // node -> neig -> col[j] -> node
            ++triangleCount;
        }
      }
    }
  }

  delete[] incoming;
  return triangleCount;
}

template <typename IDType, typename NNZType, typename ValueType>
int64_t UndirectedTriangleCount(format::CSR<IDType, NNZType, ValueType> *csr) {
  int n = csr->get_dimensions()[0];
  NNZType *row_ptr = csr->get_row_ptr();
  IDType *col = csr->get_col();

  int *isConnected = new int[n];
  for (int i = 0; i < n; ++i) {
    isConnected[i] = 0;
  }

  int64_t triangleCount = 0;
  for (int node = 0; node < n; ++node) {
    for (int i = row_ptr[node]; i < row_ptr[node + 1]; ++i) {
      isConnected[col[i]] = node;  // col[i] <-> node
    }

    for (int i = row_ptr[node]; i < row_ptr[node + 1]; ++i) {
      int neig = col[i];
      if (node < neig) {
        for (int j = row_ptr[neig]; j < row_ptr[neig + 1]; ++j) {
          if (neig < col[j] && isConnected[col[j]])  // node, neig, col[j]
            ++triangleCount;
        }
      }
    }
  }
  delete[] isConnected;
  return triangleCount;
}

template <typename IDType, typename NNZType, typename ValueType>
int64_t *TriangleCount<IDType, NNZType, ValueType>::GetTriangleCountCSR(
    std::vector<format::Format *> formats, utils::Parameters *params) {
  TriangleCountParams *param = static_cast<TriangleCountParams *>(params);
  bool countDirected = param->countDirected;
  auto csr = formats[0]->AsAbsolute<format::CSR<IDType, NNZType, ValueType>>();

  int64_t triangleCount = 0;
  if (countDirected) {
    triangleCount = DirectedTriangleCount(csr);
  } else {
    triangleCount = UndirectedTriangleCount(csr);
  }

  return new int64_t(triangleCount);
}

#if !defined(_HEADER_ONLY)
#include "init/triangle_count.inc"
#endif
}  // namespace sparsebase::feature
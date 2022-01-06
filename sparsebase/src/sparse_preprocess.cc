#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_converter.h"
#include "sparsebase/sparse_format.h"
#include <iostream>
#include <memory>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sparsebase {
std::size_t FormatVectorHash::operator()(std::vector<Format> vf) const {
  int hash = 0;
  for (auto f : vf)
    hash += f * 19381;
  return hash;
}
template <class Preprocess, typename Function, typename Key, typename KeyHash,
          typename KeyEqualTo>
bool MapToFunctionMixin<Preprocess, Function, Key, KeyHash, KeyEqualTo>::
    RegisterFunctionNoOverride(const Key &key_of_function,
                                  const Function &func_ptr) {
  if (_map_to_function.find(key_of_function) == _map_to_function.end()) {
    return false; // function already exists for this Key
  } else {
    _map_to_function[key_of_function] = func_ptr;
    return true;
  }
}

template <class Preprocess, typename Function, typename Key, typename KeyHash,
          typename KeyEqualTo>
void MapToFunctionMixin<Preprocess, Function, Key, KeyHash, KeyEqualTo>::
    RegisterFunction(const Key &key_of_function, const Function &func_ptr) {
  _map_to_function[key_of_function] = func_ptr;
}
template <class Preprocess, typename Function, typename Key, typename KeyHash,
          typename KeyEqualTo>
bool MapToFunctionMixin<Preprocess, Function, Key, KeyHash, KeyEqualTo>::
    UnregisterFunction(const Key &key_of_function) {
  if (_map_to_function.find(key_of_function) == _map_to_function.end()) {
    return false; // function already exists for this Key
  } else {
    _map_to_function.erase(key_of_function);
    return true;
  }
}
template <class Parent, typename IDType, typename NNZType, typename ValueType>
void SparseConverterMixin<Parent, IDType, NNZType, ValueType>::SetConverter(
    const SparseConverter<IDType, NNZType, ValueType> &new_sc) {
  sc_ = new_sc;
}
template <class Parent, typename IDType, typename NNZType, typename ValueType>
void SparseConverterMixin<Parent, IDType, NNZType, ValueType>::ResetConverter() {
  SparseConverter<IDType, NNZType, ValueType> new_sc;
  sc_ = new_sc;
}
template <typename IDType, typename NNZType, typename ValueType>
ReorderPreprocessType<IDType, NNZType, ValueType>::~ReorderPreprocessType(){};

template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
std::tuple<PreprocessFunction, ConversionSchema>
FormatMatcherMixin<IDType, NNZType, ValueType, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash, KeyEqualTo>::
    GetFunction(Key key, ConversionMap map,
                 SparseConverter<IDType, NNZType, ValueType>& sc) {
  ConversionSchema cs;
  PreprocessFunction func = nullptr;
  if (map.find(key) != map.end()) {
    for (auto f : key) {
      cs.push_back(std::make_tuple(false, (Format)f));
    }
    func = map[key];
  } else {
    std::vector<Key> all_keys;
    for (auto key_func : map) {
      all_keys.push_back(key_func.first);
    }
    std::vector<std::tuple<unsigned int, ConversionSchema, Key>>
        usable_keys;
    for (auto potential_key : all_keys) {
      if (potential_key.size() == key.size()) {
        ConversionSchema temp_cs;
        int conversions = 0;
        bool is_usable = true;
        for (int i = 0; i < potential_key.size(); i++) {
          if (key[i] == potential_key[i]) {
            temp_cs.push_back(std::make_tuple(false, potential_key[i]));
          } else if (sc.CanConvert(key[i], potential_key[i])) {
            temp_cs.push_back(std::make_tuple(true, potential_key[i]));
            conversions++;
          } else {
            is_usable = false;
          }
        }
        if (is_usable) {
          usable_keys.push_back(
              std::make_tuple(conversions, temp_cs, potential_key));
        }
      }
    }
    if (usable_keys.size() == 0) {
      throw 1; // TODO: add a custom exception type
    }
    std::tuple<PreprocessFunction, ConversionSchema> best_conversion;
    unsigned int num_conversions = (unsigned int)-1;
    for (auto potential_usable_key : usable_keys) {
      if (num_conversions > std::get<0>(potential_usable_key)) {
        num_conversions = std::get<0>(potential_usable_key);
        cs = std::get<1>(potential_usable_key);
        func = map[std::get<2>(potential_usable_key)];
      }
    }
  }
  return std::make_tuple(func, cs);
}
template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F>
std::vector<Format>
FormatMatcherMixin<IDType, NNZType, ValueType, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackFormats(F sf) {
  SparseFormat<IDType, NNZType, ValueType> *casted =
      static_cast<SparseFormat<IDType, NNZType, ValueType> *>(sf);
  return {casted->get_format()};
}
template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::vector<Format>
FormatMatcherMixin<IDType, NNZType, ValueType, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackFormats(F sf, SF... sfs) {
  SparseFormat<IDType, NNZType, ValueType> *casted =
      static_cast<SparseFormat<IDType, NNZType, ValueType> *>(sf);
  std::vector<Format> f = {casted->get_format()};
  std::vector<Format> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F>
std::vector<F>
FormatMatcherMixin<IDType, NNZType, ValueType, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackSFS(F sf) {
  return {sf};
}
template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::vector<F>
FormatMatcherMixin<IDType, NNZType, ValueType, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackSFS(F sf, SF... sfs) {
  std::vector<F> f = {sf};
  std::vector<F> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename IDType, typename NNZType, typename ValueType,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<PreprocessFunction, std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
FormatMatcherMixin<IDType, NNZType, ValueType, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash, KeyEqualTo>::
    Execute(ConversionMap map, SparseConverter<IDType, NNZType, ValueType>& sc, F sf,
            SF... sfs) {
  // pack the SFs into a vector
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> packed_sfs = PackSFS(sf, sfs...);
  // pack the SF formats into a vector
  std::vector<Format> formats = PackFormats(sf, sfs...);
  // get conversion schema
  std::tuple<PreprocessFunction, ConversionSchema> ret =
      GetFunction(formats, map, sc);
  PreprocessFunction func = std::get<0>(ret);
  ConversionSchema cs = std::get<1>(ret);
  // carry out conversion
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> converted =
      sc.ApplyConversionSchema(cs, packed_sfs);
  // carry out the correct call using the map
  return std::make_tuple(func, converted);
  // return std::get<0>(cs)(packed_sfs);
}
template <typename IDType, typename NNZType, typename ValueType>
GenericReorder<IDType, NNZType, ValueType>::GenericReorder() {}
template <typename IDType, typename NNZType, typename ValueType>
DegreeReorder<IDType, NNZType, ValueType>::DegreeReorder(int hyperparameter) {
  // this->map[{kCSRFormat}]= calculate_order_csr;
  this->RegisterFunction({kCSRFormat}, CalculateReorderCSR);
  this->params_ = std::unique_ptr<DegreeReorderParams>(
      new DegreeReorderParams(hyperparameter));
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *DegreeReorder<IDType, NNZType, ValueType>::CalculateReorderCSR(
    std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
    ReorderParams *params) {
  CSR<IDType, NNZType, ValueType> *csr =
      static_cast<CSR<IDType, NNZType, ValueType> *>(formats[0]);
  DegreeReorderParams *cast_params = static_cast<DegreeReorderParams *>(params);
  std::cout << cast_params->hyperparameter;
  IDType n = csr->get_dimensions()[0];
  IDType *counts = new IDType[n]();
  for (IDType u = 0; u < n; u++) {
    counts[csr->row_ptr_[u + 1] - csr->row_ptr_[u] + 1]++;
  }
  for (IDType u = 1; u < n; u++) {
    counts[u] += counts[u - 1];
  }
  IDType *sorted = new IDType[n];
  memset(sorted, -1, sizeof(IDType) * n);
  IDType *mr = new IDType[n]();
  for (IDType u = 0; u < n; u++) {
    IDType ec = counts[csr->row_ptr_[u + 1] - csr->row_ptr_[u]];
    sorted[ec + mr[ec]] = u;
    mr[ec]++;
  }
  IDType *inv_sorted = new IDType[n];
  for (IDType i = 0; i < n; i++)
    inv_sorted[sorted[i]] = i;
  delete[] mr;
  delete[] counts;
  delete[] sorted;
  return inv_sorted;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *DegreeReorderInstance<IDType, NNZType, ValueType>::GetReorder(
    SparseFormat<IDType, NNZType, ValueType> *csr) {
  std::tuple<ReorderFunction<IDType, NNZType, ValueType>,
             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  ReorderFunction<IDType, NNZType, ValueType> func = std::get<0>(func_formats);
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  return func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType>
RCMReorder<IDType, NNZType, ValueType>::RCMReorder(float a, float b) {
  this->RegisterFunction({kCSRFormat}, GetReorderCSR);
  this->params_ = std::unique_ptr<RCMReorderParams>(new RCMReorderParams(a, b));
}
template <typename IDType, typename NNZType, typename ValueType>
IDType RCMReorder<IDType, NNZType, ValueType>::peripheral(NNZType *xadj, IDType *adj, IDType n,
                                                IDType start, SignedID *distance,
                                                IDType *Q) {
  IDType r = start;
  SignedID rlevel = -1;
  SignedID qlevel = 0;

  while (rlevel != qlevel) {
    // cout << "Finding peripheral: current dist = " << qlevel << std::endl;;
    rlevel = qlevel;

    for (IDType i = 0; i < n; i++)
      distance[i] = -1;
    IDType qrp = 0, qwp = 0;
    distance[r] = 0;
    Q[qwp++] = r;

    while (qrp < qwp) {
      IDType u = Q[qrp++];
      for (NNZType ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
        IDType v = adj[ptr];
        if (distance[v] == (IDType)-1) {
          distance[v] = distance[u] + 1;
          Q[qwp++] = v;
        }
      }
    }

    qlevel = 0;
    for (IDType i = 0; i < qrp; i++) {
      if (qlevel < distance[Q[i]]) {
        qlevel = distance[Q[i]];
        r = Q[i];
      }
    }
  }
  return r;
}
template <typename IDType, typename NNZType, typename ValueType>
IDType *RCMReorder<IDType, NNZType, ValueType>::GetReorderCSR(
    std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats,
    ReorderParams *params) {
  CSR<IDType, NNZType, ValueType> *csr =
      static_cast<CSR<IDType, NNZType, ValueType> *>(formats[0]);
  RCMReorderParams *params_ = static_cast<RCMReorderParams *>(params);
  std::cout << "using the parameters " << params_->alpha << " and "
            << params_->beta << std::endl;
  NNZType *xadj = csr->get_row_ptr();
  IDType *adj = csr->get_col();
  IDType n = csr->get_dimensions()[0];
  IDType *Q = new IDType[n];

  IDType *Qp = new IDType[n];
  SignedID *distance = new SignedID[n];
  IDType *V = new IDType[n];
  for (IDType i = 0; i < n; i++)
    V[i] = 0;
  std::priority_queue<std::pair<IDType, IDType>> PQ;
  int qrp = 0, qwp = 0;
  IDType reverse = n - 1;

  for (IDType i = 0; i < n; i++) {
    if (V[i] == 0) {
      if (xadj[i] == xadj[i + 1]) {
        Q[reverse--] = i;
        V[i] = 1;
        continue;
      }

      // cout << i << std::endl;
      IDType perv = peripheral(xadj, adj, n, i, distance, Qp);
      V[perv] = 1;
      Q[qwp++] = perv;

      while (qrp < qwp) {
        IDType u = Q[qrp++];
        for (IDType ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
          IDType v = adj[ptr];
          if (V[v] == 0) {
            PQ.push(std::make_pair(xadj[v + 1] - xadj[v], v));
            V[v] = 1;
          }
        }

        while (!PQ.empty()) {
          Q[qwp++] = PQ.top().second;
          ;
          PQ.pop();
        }
      }
    }
  }

  // Reverse
  for (IDType i = 0; i < n / 2; i++) {
    IDType t = Q[i];
    Q[i] = Q[n - i - 1];
    Q[n - i - 1] = t;
  }
  delete[] Qp;
  delete[] distance;
  delete[] V;
  return Q;
}
template <typename IDType, typename NNZType, typename ValueType,
          template <typename, typename, typename> class ReorderImpl>
IDType *ReorderInstance<IDType, NNZType, ValueType, ReorderImpl>::GetReorder(
    SparseFormat<IDType, NNZType, ValueType> *csr) {
  std::tuple<ReorderFunction<IDType, NNZType, ValueType>,
             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  ReorderFunction<IDType, NNZType, ValueType> func = std::get<0>(func_formats);
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  return func(sfs, this->params_.get());
}
template <typename IDType, typename NNZType, typename ValueType,
          template <typename, typename, typename> class ReorderImpl>
IDType *ReorderInstance<IDType, NNZType, ValueType, ReorderImpl>::GetReorder(
    SparseFormat<IDType, NNZType, ValueType> *csr, ReorderParams *params) {
  std::tuple<ReorderFunction<IDType, NNZType, ValueType>,
             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  ReorderFunction<IDType, NNZType, ValueType> func = std::get<0>(func_formats);
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  return func(sfs, params);
}

template <typename IDType, typename NNZType, typename ValueType>
Transform<IDType, NNZType, ValueType>::Transform(){
  this->RegisterFunction({kCSRFormat}, TransformCSR);
}
template <typename IDType, typename NNZType, typename ValueType>
TransformPreprocessType<IDType, NNZType, ValueType>::~TransformPreprocessType(){};
template <typename IDType, typename NNZType, typename ValueType>
SparseFormat<IDType, NNZType, ValueType> *Transform<IDType, NNZType, ValueType>::TransformCSR(
    std::vector<SparseFormat<IDType, NNZType, ValueType> *> formats, IDType *order) {
  SparseFormat<IDType, NNZType, ValueType> *sp = formats[0];
  std::vector<IDType> dimensions = sp->get_dimensions();
  IDType n = dimensions[0];
  IDType m = dimensions[1];
  NNZType nnz = sp->get_num_nnz();
  NNZType *xadj = sp->get_row_ptr();
  IDType *adj = sp->get_col();
  ValueType *vals = sp->get_vals();
  NNZType *nxadj = new NNZType[n + 1]();
  IDType *nadj = new IDType[nnz]();
  ValueType *nvals = nullptr;
  if constexpr (!std::is_same_v<void, ValueType>) {
    nvals = new ValueType[nnz]();
  }

  IDType *inverse_order = new IDType[n]();
  for (IDType i = 0; i < n; i++)
    inverse_order[order[i]] = i;
  NNZType c = 0;
  for (IDType i = 0; i < n; i++) {
    IDType u = inverse_order[i];
    nxadj[i + 1] = nxadj[i] + (xadj[u + 1] - xadj[u]);
    for (NNZType v = xadj[u]; v < xadj[u + 1]; v++) {
      nadj[c] = order[adj[v]];
      if constexpr (!std::is_same_v<void, ValueType>) {
        nvals[c] = vals[v];
      }
      c++;
    }
  }
  delete[] inverse_order;
  CSR<IDType, NNZType, ValueType> *csr = new CSR(n, m, nxadj, nadj, nvals);
  return csr;
}
template <typename IDType, typename NNZType, typename ValueType,
          template <typename, typename, typename> class TransformImpl>
SparseFormat<IDType, NNZType, ValueType> *
TransformInstance<IDType, NNZType, ValueType, TransformImpl>::GetTransformation(
    SparseFormat<IDType, NNZType, ValueType> *csr, IDType *ordr) {
  std::tuple<TransformFunction<IDType, NNZType, ValueType>,
             std::vector<SparseFormat<IDType, NNZType, ValueType> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  TransformFunction<IDType, NNZType, ValueType> func = std::get<0>(func_formats);
  std::vector<SparseFormat<IDType, NNZType, ValueType> *> sfs = std::get<1>(func_formats);
  return func(sfs, ordr);
}

#ifdef NDEBUG
#include "init/sparse_preprocess.inc"
#include "init/sparse_preprocess_instance.inc"
#else
template class ReorderPreprocessType<unsigned int, unsigned int, void>;

template class DegreeReorder<unsigned int, unsigned int, void>;
template class DegreeReorderInstance<unsigned int, unsigned int, void>;
template class DegreeReorderInstance<unsigned int, unsigned int, unsigned int>;
template class ReorderInstance<unsigned int, unsigned int, void, DegreeReorder>;

template class GenericReorder<unsigned int, unsigned int, void>;
template class ReorderInstance<unsigned int, unsigned int, void,
                               GenericReorder>;

template class RCMReorder<unsigned int, unsigned int, void>;
template class RCMReorderInstance<unsigned int, unsigned int, void>;
template class ReorderInstance<unsigned int, unsigned int, void, RCMReorder>;

template class Transform<unsigned int, unsigned int, void>;
template class TransformPreprocessType<unsigned int, unsigned int, void>;
template class TransformInstance<unsigned int, unsigned int, void, Transform>;
#endif
} // namespace sparsebase

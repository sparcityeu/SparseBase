#include "sparsebase/sparse_preprocess.hpp"
#include "sparsebase/sparse_converter.hpp"
#include "sparsebase/sparse_format.hpp"
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
template <class Parent, typename ID, typename NumNonZeros, typename Value>
void SparseConverterMixin<Parent, ID, NumNonZeros, Value>::SetConverter(
    const SparseConverter<ID, NumNonZeros, Value> &new_sc) {
  sc_ = new_sc;
}
template <class Parent, typename ID, typename NumNonZeros, typename Value>
void SparseConverterMixin<Parent, ID, NumNonZeros, Value>::ResetConverter() {
  SparseConverter<ID, NumNonZeros, Value> new_sc;
  sc_ = new_sc;
}
template <typename ID, typename NumNonZeros, typename Value>
ReorderPreprocessType<ID, NumNonZeros, Value>::~ReorderPreprocessType(){};

template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
std::tuple<PreprocessFunction, ConversionSchema>
FormatMatcherMixin<ID, NumNonZeros, Value, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash, KeyEqualTo>::
    GetFunction(Key key, ConversionMap map,
                 SparseConverter<ID, NumNonZeros, Value> sc) {
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
template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F>
std::vector<Format>
FormatMatcherMixin<ID, NumNonZeros, Value, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackFormats(F sf) {
  SparseFormat<ID, NumNonZeros, Value> *casted =
      static_cast<SparseFormat<ID, NumNonZeros, Value> *>(sf);
  return {casted->get_format()};
}
template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::vector<Format>
FormatMatcherMixin<ID, NumNonZeros, Value, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackFormats(F sf, SF... sfs) {
  SparseFormat<ID, NumNonZeros, Value> *casted =
      static_cast<SparseFormat<ID, NumNonZeros, Value> *>(sf);
  std::vector<Format> f = {casted->get_format()};
  std::vector<Format> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F>
std::vector<F>
FormatMatcherMixin<ID, NumNonZeros, Value, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackSFS(F sf) {
  return {sf};
}
template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::vector<F>
FormatMatcherMixin<ID, NumNonZeros, Value, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash,
                   KeyEqualTo>::PackSFS(F sf, SF... sfs) {
  std::vector<F> f = {sf};
  std::vector<F> remainder = PackFormats(sfs...);
  for (auto i : remainder) {
    f.push_back(i);
  }
  return f;
}
template <typename ID, typename NumNonZeros, typename Value,
          class PreprocessingImpl, typename PreprocessFunction,
          typename Key, typename KeyHash,
          typename KeyEqualTo>
template <typename F, typename... SF>
std::tuple<PreprocessFunction, std::vector<SparseFormat<ID, NumNonZeros, Value> *>>
FormatMatcherMixin<ID, NumNonZeros, Value, PreprocessingImpl, PreprocessFunction,
                   Key, KeyHash, KeyEqualTo>::
    Execute(ConversionMap map, SparseConverter<ID, NumNonZeros, Value> sc, F sf,
            SF... sfs) {
  // pack the SFs into a vector
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> packed_sfs = PackSFS(sf, sfs...);
  // pack the SF formats into a vector
  std::vector<Format> formats = PackFormats(sf, sfs...);
  // get conversion schema
  std::tuple<PreprocessFunction, ConversionSchema> ret =
      GetFunction(formats, map, sc);
  PreprocessFunction func = std::get<0>(ret);
  ConversionSchema cs = std::get<1>(ret);
  // carry out conversion
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> converted =
      sc.ApplyConversionSchema(cs, packed_sfs);
  // carry out the correct call using the map
  return std::make_tuple(func, converted);
  // return std::get<0>(cs)(packed_sfs);
}
template <typename ID, typename NumNonZeros, typename Value>
GenericReorder<ID, NumNonZeros, Value>::GenericReorder() {}
template <typename ID, typename NumNonZeros, typename Value>
DegreeReorder<ID, NumNonZeros, Value>::DegreeReorder(int hyperparameter) {
  // this->map[{kCSRFormat}]= calculate_order_csr;
  this->RegisterFunction({kCSRFormat}, CalculateReorderCSR);
  this->params_ = std::unique_ptr<DegreeReorderParams>(
      new DegreeReorderParams(hyperparameter));
}
template <typename ID, typename NumNonZeros, typename Value>
ID *DegreeReorder<ID, NumNonZeros, Value>::CalculateReorderCSR(
    std::vector<SparseFormat<ID, NumNonZeros, Value> *> formats,
    ReorderParams *params) {
  CSR<ID, NumNonZeros, Value> *csr =
      static_cast<CSR<ID, NumNonZeros, Value> *>(formats[0]);
  DegreeReorderParams *cast_params = static_cast<DegreeReorderParams *>(params);
  std::cout << cast_params->hyperparameter;
  ID n = csr->get_dimensions()[0];
  ID *counts = new ID[n]();
  for (ID u = 0; u < n; u++) {
    counts[csr->row_ptr_[u + 1] - csr->row_ptr_[u] + 1]++;
  }
  for (ID u = 1; u < n; u++) {
    counts[u] += counts[u - 1];
  }
  ID *sorted = new ID[n];
  memset(sorted, -1, sizeof(ID) * n);
  ID *mr = new ID[n]();
  for (ID u = 0; u < n; u++) {
    ID ec = counts[csr->row_ptr_[u + 1] - csr->row_ptr_[u]];
    sorted[ec + mr[ec]] = u;
    mr[ec]++;
  }
  ID *inv_sorted = new ID[n];
  for (ID i = 0; i < n; i++)
    inv_sorted[sorted[i]] = i;
  delete[] mr;
  delete[] counts;
  delete[] sorted;
  return inv_sorted;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *DegreeReorderInstance<ID, NumNonZeros, Value>::GetReorder(
    SparseFormat<ID, NumNonZeros, Value> *csr) {
  std::tuple<ReorderFunction<ID, NumNonZeros, Value>,
             std::vector<SparseFormat<ID, NumNonZeros, Value> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  ReorderFunction<ID, NumNonZeros, Value> func = std::get<0>(func_formats);
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> sfs = std::get<1>(func_formats);
  return func(sfs, this->params_.get());
}
template <typename ID, typename NumNonZeros, typename Value>
RCMReorder<ID, NumNonZeros, Value>::RCMReorder(float a, float b) {
  this->RegisterFunction({kCSRFormat}, GetReorderCSR);
  this->params_ = std::unique_ptr<RCMReorderParams>(new RCMReorderParams(a, b));
}
template <typename ID, typename NumNonZeros, typename Value>
ID RCMReorder<ID, NumNonZeros, Value>::peripheral(NumNonZeros *xadj, ID *adj, ID n,
                                                ID start, SignedID *distance,
                                                ID *Q) {
  ID r = start;
  SignedID rlevel = -1;
  SignedID qlevel = 0;

  while (rlevel != qlevel) {
    // cout << "Finding peripheral: current dist = " << qlevel << std::endl;;
    rlevel = qlevel;

    for (ID i = 0; i < n; i++)
      distance[i] = -1;
    ID qrp = 0, qwp = 0;
    distance[r] = 0;
    Q[qwp++] = r;

    while (qrp < qwp) {
      ID u = Q[qrp++];
      for (NumNonZeros ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
        ID v = adj[ptr];
        if (distance[v] == (ID)-1) {
          distance[v] = distance[u] + 1;
          Q[qwp++] = v;
        }
      }
    }

    qlevel = 0;
    for (ID i = 0; i < qrp; i++) {
      if (qlevel < distance[Q[i]]) {
        qlevel = distance[Q[i]];
        r = Q[i];
      }
    }
  }
  return r;
}
template <typename ID, typename NumNonZeros, typename Value>
ID *RCMReorder<ID, NumNonZeros, Value>::GetReorderCSR(
    std::vector<SparseFormat<ID, NumNonZeros, Value> *> formats,
    ReorderParams *params) {
  CSR<ID, NumNonZeros, Value> *csr =
      static_cast<CSR<ID, NumNonZeros, Value> *>(formats[0]);
  RCMReorderParams *params_ = static_cast<RCMReorderParams *>(params);
  std::cout << "using the parameters " << params_->alpha << " and "
            << params_->beta << std::endl;
  NumNonZeros *xadj = csr->get_row_ptr();
  ID *adj = csr->get_col();
  ID n = csr->get_dimensions()[0];
  ID *Q = new ID[n];

  ID *Qp = new ID[n];
  SignedID *distance = new SignedID[n];
  ID *V = new ID[n];
  for (ID i = 0; i < n; i++)
    V[i] = 0;
  std::priority_queue<std::pair<ID, ID>> PQ;
  int qrp = 0, qwp = 0;
  ID reverse = n - 1;

  for (ID i = 0; i < n; i++) {
    if (V[i] == 0) {
      if (xadj[i] == xadj[i + 1]) {
        Q[reverse--] = i;
        V[i] = 1;
        continue;
      }

      // cout << i << std::endl;
      ID perv = peripheral(xadj, adj, n, i, distance, Qp);
      V[perv] = 1;
      Q[qwp++] = perv;

      while (qrp < qwp) {
        ID u = Q[qrp++];
        for (ID ptr = xadj[u]; ptr < xadj[u + 1]; ptr++) {
          ID v = adj[ptr];
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
  for (ID i = 0; i < n / 2; i++) {
    ID t = Q[i];
    Q[i] = Q[n - i - 1];
    Q[n - i - 1] = t;
  }
  delete[] Qp;
  delete[] distance;
  delete[] V;
  return Q;
}
template <typename ID, typename NumNonZeros, typename Value,
          template <typename, typename, typename> class ReorderImpl>
ID *ReorderInstance<ID, NumNonZeros, Value, ReorderImpl>::GetReorder(
    SparseFormat<ID, NumNonZeros, Value> *csr) {
  std::tuple<ReorderFunction<ID, NumNonZeros, Value>,
             std::vector<SparseFormat<ID, NumNonZeros, Value> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  ReorderFunction<ID, NumNonZeros, Value> func = std::get<0>(func_formats);
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> sfs = std::get<1>(func_formats);
  return func(sfs, this->params_.get());
}
template <typename ID, typename NumNonZeros, typename Value,
          template <typename, typename, typename> class ReorderImpl>
ID *ReorderInstance<ID, NumNonZeros, Value, ReorderImpl>::GetReorder(
    SparseFormat<ID, NumNonZeros, Value> *csr, ReorderParams *params) {
  std::tuple<ReorderFunction<ID, NumNonZeros, Value>,
             std::vector<SparseFormat<ID, NumNonZeros, Value> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  ReorderFunction<ID, NumNonZeros, Value> func = std::get<0>(func_formats);
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> sfs = std::get<1>(func_formats);
  return func(sfs, params);
}

template <typename ID, typename NumNonZeros, typename Value>
Transform<ID, NumNonZeros, Value>::Transform(){
  this->RegisterFunction({kCSRFormat}, TransformCSR);
}
template <typename ID, typename NumNonZeros, typename Value>
TransformPreprocessType<ID, NumNonZeros, Value>::~TransformPreprocessType(){};
template <typename ID, typename NumNonZeros, typename Value>
SparseFormat<ID, NumNonZeros, Value> *Transform<ID, NumNonZeros, Value>::TransformCSR(
    std::vector<SparseFormat<ID, NumNonZeros, Value> *> formats, ID *order) {
  SparseFormat<ID, NumNonZeros, Value> *sp = formats[0];
  std::vector<ID> dimensions = sp->get_dimensions();
  ID n = dimensions[0];
  ID m = dimensions[1];
  NumNonZeros nnz = sp->get_num_nnz();
  NumNonZeros *xadj = sp->get_row_ptr();
  ID *adj = sp->get_col();
  Value *vals = sp->get_vals();
  NumNonZeros *nxadj = new ID[n + 1]();
  ID *nadj = new NumNonZeros[nnz]();
  Value *nvals;
  if constexpr (!std::is_same_v<void, Value>) {
    nvals = new Value[nnz]();
  }

  ID *inverse_order = new ID[n]();
  for (ID i = 0; i < n; i++)
    inverse_order[order[i]] = i;
  NumNonZeros c = 0;
  for (ID i = 0; i < n; i++) {
    ID u = inverse_order[i];
    nxadj[i + 1] = nxadj[i] + (xadj[u + 1] - xadj[u]);
    for (NumNonZeros v = xadj[u]; v < xadj[u + 1]; v++) {
      nadj[c] = order[adj[v]];
      if constexpr (!std::is_same_v<void, Value>) {
        nvals[c] = vals[v];
      }
      c++;
    }
  }
  delete[] inverse_order;
  CSR<ID, NumNonZeros, Value> *csr = new CSR(n, m, nxadj, nadj, nvals);
  return csr;
}
template <typename ID, typename NumNonZeros, typename Value,
          template <typename, typename, typename> class TransformImpl>
SparseFormat<ID, NumNonZeros, Value> *
TransformInstance<ID, NumNonZeros, Value, TransformImpl>::GetTransformation(
    SparseFormat<ID, NumNonZeros, Value> *csr, ID *ordr) {
  std::tuple<TransformFunction<ID, NumNonZeros, Value>,
             std::vector<SparseFormat<ID, NumNonZeros, Value> *>>
      func_formats = this->Execute(this->_map_to_function, this->sc_, csr);
  TransformFunction<ID, NumNonZeros, Value> func = std::get<0>(func_formats);
  std::vector<SparseFormat<ID, NumNonZeros, Value> *> sfs = std::get<1>(func_formats);
  return func(sfs, ordr);
}
template class DegreeReorder<unsigned int, unsigned int, void>;
template class ReorderPreprocessType<unsigned int, unsigned int, void>;

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
} // namespace sparsebase

#include "sparsebase/SparsePreprocess.hpp"
#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseConverter.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <queue>
#include <utility>
#include <memory>

namespace sparsebase {
    std::size_t FormatVectorHash::operator()(std::vector<Format> vf) const{
      int hash = 0;
      for (auto f : vf) hash+=f*19381; 
      return hash;
    }
  template <class Preprocess, typename func, typename key, typename key_hash, typename key_equal>
    bool MapToFunctionMixin<Preprocess, func, key, key_hash, key_equal>::register_function_no_override(const key& key_of_function, const func & func_ptr){
      if (_map_to_function.find(key_of_function) == _map_to_function.end()){
        return false;  // function already exists for this key
      } else {
        _map_to_function[key_of_function] = func_ptr;
        return true;
      }
    }

  template <class Preprocess, typename func, typename key, typename key_hash, typename key_equal>
    void MapToFunctionMixin<Preprocess, func, key, key_hash, key_equal>::register_function(const key& key_of_function, const func & func_ptr){
      _map_to_function[key_of_function] = func_ptr;
    }
  template <class Preprocess, typename func, typename key, typename key_hash, typename key_equal>
    bool MapToFunctionMixin<Preprocess, func, key, key_hash, key_equal>::unregister_function(const key& key_of_function){
      if (_map_to_function.find(key_of_function) == _map_to_function.end()){
        return false;  // function already exists for this key
      } else {
        _map_to_function.erase(key_of_function);
        return true;
      }
    }
  template <class Parent, typename ID_t, typename NNZ_t, typename VAL_t>
      void SparseConverterMixin<Parent, ID_t, NNZ_t, VAL_t>::set_converter(const SparseConverter<ID_t, NNZ_t, VAL_t> & new_sc){
        _sc = new_sc;
      }
  template <class Parent, typename ID_t, typename NNZ_t, typename VAL_t>
      void SparseConverterMixin<Parent, ID_t, NNZ_t, VAL_t>::reset_converter(){
        SparseConverter<ID_t, NNZ_t, VAL_t> new_sc;
        _sc = new_sc;
      }
  template<typename ID_t, typename NNZ_t, typename VAL_t>
      ReorderPreprocessType<ID_t, NNZ_t, VAL_t>::~ReorderPreprocessType(){};

  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key, typename config_key_hash, typename config_key_equal_to>
    std::tuple<PreprocessFunction, conversion_schema> FormatMatcherMixin<ID_t, NNZ_t, VAL_t, PreprocessingImpl, PreprocessFunction, config_key,config_key_hash, config_key_equal_to>::get_function(config_key key, conversion_map map, SparseConverter<ID_t, NNZ_t, VAL_t> sc){
      conversion_schema cs;
      PreprocessFunction func = nullptr;
      if (map.find(key) != map.end()){
        for (auto f : key){
          cs.push_back(make_tuple(false, (Format)f));
        }
        func = map[key];
      } 
      else {
        std::vector<config_key> all_keys;
        for (auto key_func : map){
          all_keys.push_back(key_func.first);
        } 
        std::vector<std::tuple<unsigned int, conversion_schema, config_key>> usable_keys;
        for (auto potential_key : all_keys){
          if (potential_key.size() == key.size()){
            conversion_schema temp_cs;
            int conversions = 0;
            bool is_usable = true;
            for (int i =0; i < potential_key.size(); i++){
              if (key[i] == potential_key[i]){
                temp_cs.push_back(make_tuple(false, potential_key[i]));
              }
              else if (sc.can_convert(key[i], potential_key[i])){
                temp_cs.push_back(make_tuple(true, potential_key[i]));
                conversions++;
              } else {
                is_usable = false;
              }
            }
            if (is_usable){
              usable_keys.push_back(make_tuple(conversions,temp_cs,potential_key));
            }
          }
        }
        if (usable_keys.size() == 0){
          throw 1; // TODO: add a custom exception type
        }
        std::tuple<PreprocessFunction, conversion_schema> best_conversion;
        unsigned int num_conversions = (unsigned int)-1;
        for (auto potential_usable_key : usable_keys){
          if (num_conversions > get<0>(potential_usable_key)){
            num_conversions = get<0>(potential_usable_key);
            cs = get<1>(potential_usable_key);
            func = map[get<2>(potential_usable_key)];
          }
        }
      }
      return make_tuple(func, cs);
    }
  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key, typename config_key_hash, typename config_key_equal_to>
    template <typename F>
    std::vector<Format> FormatMatcherMixin<ID_t, NNZ_t, VAL_t, PreprocessingImpl, PreprocessFunction, config_key,config_key_hash, config_key_equal_to>::pack_formats(F sf){
      SparseFormat<ID_t, NNZ_t, VAL_t>* casted = static_cast<SparseFormat<ID_t, NNZ_t, VAL_t>*>(sf);
      return {casted->get_format()};
    }
  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key, typename config_key_hash, typename config_key_equal_to> template <typename F, typename... SF>
    std::vector<Format> FormatMatcherMixin<ID_t, NNZ_t, VAL_t, PreprocessingImpl, PreprocessFunction, config_key,config_key_hash, config_key_equal_to>::pack_formats(F sf, SF... sfs){
      SparseFormat<ID_t, NNZ_t, VAL_t>* casted = static_cast<SparseFormat<ID_t, NNZ_t, VAL_t>*>(sf);
      std::vector<Format> f = {casted->get_format()};
      std::vector<Format> remainder = pack_formats(sfs...);
      for (auto i : remainder){
        f.push_back(i);
      }
      return f;
    }
  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key, typename config_key_hash, typename config_key_equal_to> template <typename F>
    std::vector<F> FormatMatcherMixin<ID_t, NNZ_t, VAL_t, PreprocessingImpl, PreprocessFunction, config_key,config_key_hash, config_key_equal_to>::pack_sfs(F sf){
      return {sf};
    }
  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key, typename config_key_hash, typename config_key_equal_to> template <typename F, typename... SF>
    std::vector<F> FormatMatcherMixin<ID_t, NNZ_t, VAL_t, PreprocessingImpl, PreprocessFunction, config_key,config_key_hash, config_key_equal_to>::pack_sfs(F sf, SF... sfs){
      std::vector<F> f = {sf};
      std::vector<F> remainder = pack_formats(sfs...);
      for (auto i : remainder){
        f.push_back(i);
      }
      return f;
    }
  template <typename ID_t, typename NNZ_t, typename VAL_t, class PreprocessingImpl, typename PreprocessFunction, typename config_key, typename config_key_hash, typename config_key_equal_to> template <typename F, typename... SF>
    std::tuple<PreprocessFunction, std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*>> FormatMatcherMixin<ID_t, NNZ_t, VAL_t, PreprocessingImpl, PreprocessFunction, config_key,config_key_hash, config_key_equal_to>::execute(conversion_map map, SparseConverter<ID_t, NNZ_t, VAL_t> sc, F sf, SF... sfs){
      // pack the SFs into a vector
      vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> packed_sfs = pack_sfs(sf, sfs...);
      // pack the SF formats into a vector
      vector<Format> formats = pack_formats(sf, sfs...);
      // get conversion schema
      std::tuple<PreprocessFunction, conversion_schema> ret = get_function(formats, map, sc);
      PreprocessFunction func = get<0>(ret);
      conversion_schema cs = get<1>(ret);
      // carry out conversion
      std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> converted = sc.apply_conversion_schema(cs, packed_sfs);
      // carry out the correct call using the map
      return make_tuple(func, converted);
      //return get<0>(cs)(packed_sfs);
    }
  template<typename ID_t, typename NNZ_t, typename VAL_t>
      DegreeReorder<ID_t, NNZ_t, VAL_t>::DegreeReorder(int hyperparameter){
        //this->map[{CSR_f}]= calculate_order_csr;
        this->register_function({CSR_f}, calculate_Reorder_csr);
        this->_params = std::unique_ptr<DegreeReorderParams>(new DegreeReorderParams(hyperparameter));
      }
  template<typename ID_t, typename NNZ_t, typename VAL_t>
      ID_t* DegreeReorder<ID_t, NNZ_t, VAL_t>::calculate_Reorder_csr(std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> formats, ReorderParams* params){
        CSR<ID_t, NNZ_t, VAL_t>* csr = static_cast<CSR<ID_t, NNZ_t, VAL_t>*>(formats[0]);
        DegreeReorderParams* cast_params = static_cast<DegreeReorderParams*>(params);
        cout <<cast_params->_hyperparameter;
        ID_t n = csr->get_dimensions()[0];
        ID_t * counts = new ID_t[n]();
        for(ID_t u = 0; u < n; u++){
          counts[csr->xadj[u+1] - csr->xadj[u]+1]++;
        }
        for(ID_t u = 1; u < n; u++){
          counts[u] += counts[u - 1];
        }
        ID_t * sorted = new ID_t[n];
        memset(sorted, -1, sizeof(ID_t) * n);
        ID_t * mr = new ID_t[n]();
        for(ID_t u = 0; u < n; u++){
          ID_t ec = counts[csr->xadj[u+1] - csr->xadj[u]];
          sorted[ec + mr[ec]] = u;
          mr[ec]++;
        }
        ID_t * inv_sorted = new ID_t[n];
        for (ID_t i = 0; i < n; i++) inv_sorted[sorted[i]] = i;
        delete [] mr;
        delete [] counts;
        delete [] sorted;
        return inv_sorted;
      }
  template <typename ID_t, typename NNZ_t, typename VAL_t>
    ID_t* DegreeReorderInstance<ID_t, NNZ_t, VAL_t>::get_reorder(SparseFormat<ID_t, NNZ_t, VAL_t>* csr){
      std::tuple <ReorderFunction<ID_t, NNZ_t, VAL_t>, std::vector<SparseFormat<ID_t, NNZ_t, VAL_t> *>> func_formats = this->execute(this->_map_to_function, this->_sc, csr);
      ReorderFunction<ID_t, NNZ_t, VAL_t> func = get<0>(func_formats);
      std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> sfs = get<1>(func_formats);
      return func(sfs, this->_params.get());
    }
  template<typename ID_t, typename NNZ_t, typename VAL_t>
      RCMReorder<ID_t, NNZ_t, VAL_t>::RCMReorder(float a, float b) {
        this->register_function({CSR_f}, get_reorder_csr);
        this->_params = std::unique_ptr<RCMReorderParams>(new RCMReorderParams(a, b));
      }
      template <typename ID_t, typename NNZ_t, typename VAL_t>
      ID_t RCMReorder<ID_t, NNZ_t, VAL_t>::peripheral(NNZ_t *xadj, ID_t *adj, ID_t n, ID_t start, s_ID_t *distance, ID_t *Q)
      {
        ID_t r = start;
        s_ID_t rlevel = -1;
        s_ID_t qlevel = 0;

        while (rlevel != qlevel)
        {
          // cout << "Finding peripheral: current dist = " << qlevel << endl;;
          rlevel = qlevel;

          for (ID_t i = 0; i < n; i++)
            distance[i] = -1;
          ID_t qrp = 0, qwp = 0;
          distance[r] = 0;
          Q[qwp++] = r;

          while (qrp < qwp)
          {
            ID_t u = Q[qrp++];
            for (NNZ_t ptr = xadj[u]; ptr < xadj[u + 1]; ptr++)
            {
              ID_t v = adj[ptr];
              if (distance[v] == (ID_t)-1)
              {
                distance[v] = distance[u] + 1;
                Q[qwp++] = v;
              }
            }
          }

          qlevel = 0;
          for (ID_t i = 0; i < qrp; i++)
          {
            if (qlevel < distance[Q[i]])
            {
              qlevel = distance[Q[i]];
              r = Q[i];
            }
          }
        }
        return r;
      }
      template <typename ID_t, typename NNZ_t, typename VAL_t>
      ID_t *RCMReorder<ID_t, NNZ_t, VAL_t>::get_reorder_csr(std::vector<SparseFormat<ID_t, NNZ_t, VAL_t> *> formats, ReorderParams* params)
      {
        CSR<ID_t, NNZ_t, VAL_t>* csr = static_cast<CSR<ID_t, NNZ_t, VAL_t>*>(formats[0]);
        RCMReorderParams* _params = static_cast<RCMReorderParams*>(params); 
        std::cout << "using the parameters " << _params->alpha << " and " << _params->beta << std::endl;
        NNZ_t * xadj = csr->get_xadj();
        ID_t * adj = csr->get_adj();
        ID_t n = csr->get_dimensions()[0];
        ID_t *Q = new ID_t[n];

        ID_t *Qp = new ID_t[n];
        s_ID_t *distance = new s_ID_t[n];
        ID_t *V = new ID_t[n];
        for (ID_t i = 0; i < n; i++)
          V[i] = 0;
        std::priority_queue<pair<ID_t, ID_t> > PQ;
        int qrp = 0, qwp = 0;
        ID_t reverse = n - 1;

        for (ID_t i = 0; i < n; i++)
        {
          if (V[i] == 0)
          {
            if (xadj[i] == xadj[i + 1])
            {
              Q[reverse--] = i;
              V[i] = 1;
              continue;
            }

            // cout << i << endl;
            ID_t perv = peripheral(xadj, adj, n, i, distance, Qp);
            V[perv] = 1;
            Q[qwp++] = perv;

            while (qrp < qwp)
            {
              ID_t u = Q[qrp++];
              for (ID_t ptr = xadj[u]; ptr < xadj[u + 1]; ptr++)
              {
                ID_t v = adj[ptr];
                if (V[v] == 0)
                {
                  PQ.push(std::make_pair(xadj[v + 1] - xadj[v], v));
                  V[v] = 1;
                }
              }

              while (!PQ.empty())
              {
                Q[qwp++] = PQ.top().second;
                ;
                PQ.pop();
              }
            }
          }
        }

        // Reverse
        for (ID_t i = 0; i < n / 2; i++)
        {
          ID_t t = Q[i];
          Q[i] = Q[n - i - 1];
          Q[n - i - 1] = t;
        }
        delete [] Qp;
        delete [] distance;
        delete [] V;
        return Q;
      }
  template <typename ID_t, typename NNZ_t, typename VAL_t, template<typename, typename, typename> class Reorder_T>
    ID_t* ReorderInstance<ID_t, NNZ_t, VAL_t, Reorder_T>::get_reorder(SparseFormat<ID_t, NNZ_t, VAL_t>* csr){
      std::tuple <ReorderFunction<ID_t, NNZ_t, VAL_t>, std::vector<SparseFormat<ID_t, NNZ_t, VAL_t> *>> func_formats = this->execute(this->_map_to_function, this->_sc, csr);
      ReorderFunction<ID_t, NNZ_t, VAL_t> func = get<0>(func_formats);
      std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> sfs = get<1>(func_formats);
      return func(sfs, this->_params.get());
    }

  template<typename ID_t, typename NNZ_t, typename VAL_t>
    Transform<ID_t, NNZ_t, VAL_t>::Transform(int hyperparameter):_hyperparameter(hyperparameter){
      this->register_function({CSR_f}, transform_csr);
    }
  template<typename ID_t, typename NNZ_t, typename VAL_t>
      TransformPreprocessType<ID_t, NNZ_t, VAL_t>::~TransformPreprocessType(){};
  template<typename ID_t, typename NNZ_t, typename VAL_t>
    SparseFormat<ID_t, NNZ_t, VAL_t>* Transform<ID_t, NNZ_t, VAL_t>::transform_csr(std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> formats, ID_t * order){
      SparseFormat<ID_t, NNZ_t, VAL_t>* sp = formats[0];
      vector<ID_t> dimensions = sp->get_dimensions();
      ID_t n = dimensions[0];
      ID_t m = dimensions[1];
      NNZ_t nnz = sp->get_num_nnz();
      NNZ_t * xadj = sp->get_xadj();
      ID_t * adj = sp->get_adj();
      VAL_t * vals = sp->get_vals();
      NNZ_t * nxadj = new ID_t[n+1]();
      ID_t * nadj = new NNZ_t[nnz]();
      VAL_t * nvals;
      if constexpr (!std::is_same_v<void, VAL_t>){
        nvals = new VAL_t[nnz]();
      }

      ID_t * inverse_order = new ID_t[n]();
      for (ID_t i = 0; i < n; i++) inverse_order[order[i]] = i;
      NNZ_t c = 0;
      for (ID_t i = 0; i < n; i++){
        ID_t u = inverse_order[i];
        nxadj[i+1] = nxadj[i] + (xadj[u+1] - xadj[u]); 
        for (NNZ_t v = xadj[u]; v < xadj[u+1]; v++){
          nadj[c] = order[adj[v]];
          if constexpr (!std::is_same_v<void, VAL_t>){
            nvals[c] = vals[v];
          }
          c++;
        }
      }
      delete [] inverse_order;
      CSR<ID_t, NNZ_t, VAL_t> * csr = new CSR(n, m, nxadj, nadj, nvals);
      return csr;
    }
  template <typename ID_t, typename NNZ_t, typename VAL_t, template<typename, typename, typename> class TRANSFORM_t>
    SparseFormat<ID_t, NNZ_t, VAL_t>* TransformInstance<ID_t, NNZ_t, VAL_t, TRANSFORM_t>::get_transformation(SparseFormat<ID_t, NNZ_t, VAL_t>* csr, ID_t * ordr){
      std::tuple <TransformFunction<ID_t, NNZ_t, VAL_t>, std::vector<SparseFormat<ID_t, NNZ_t, VAL_t> *>> func_formats = this->execute(this->_map_to_function, this->_sc, csr);
      TransformFunction<ID_t, NNZ_t, VAL_t> func = get<0>(func_formats);
      std::vector<SparseFormat<ID_t, NNZ_t, VAL_t>*> sfs = get<1>(func_formats);
      return func(sfs, ordr);
    }
    template class DegreeReorder<unsigned int, unsigned int, void>;
    template class ReorderPreprocessType<unsigned int, unsigned int, void>;

    template class DegreeReorderInstance<unsigned int, unsigned int, void>;
    template class DegreeReorderInstance<unsigned int, unsigned int, unsigned int>;
    template class ReorderInstance<unsigned int, unsigned int, void, DegreeReorder>;

    template class RCMReorder<unsigned int, unsigned int, void>;
    template class RCMReorderInstance<unsigned int, unsigned int, void>;
    template class ReorderInstance<unsigned int, unsigned int, void, RCMReorder>;

    template class Transform<unsigned int, unsigned int, void>;
    template class TransformPreprocessType<unsigned int, unsigned int, void>;
    template class TransformInstance<unsigned int, unsigned int, void, Transform>;
}

#include "sparsebase/io/edge_list_writer.h"

#include <string>

#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
EdgeListWriter<IDType, NNZType, ValueType>::EdgeListWriter(
    std::string filename, bool directed)
    : filename_(filename),
      directed_(directed) {}

template <typename IDType, typename NNZType, typename ValueType>
void EdgeListWriter<IDType, NNZType, ValueType>::WriteCOO(
    format::COO<IDType, NNZType, ValueType> *coo) const {
  std::ofstream edgeListFile;
  edgeListFile.open(filename_);
  if constexpr (std::is_same_v<ValueType, void>) {
    IDType* row = coo->get_row();
    IDType* col = coo->get_col();
    std::vector<std::tuple<IDType, IDType>> edges;
    for (int i = 0; i < coo->get_num_nnz(); ++i) {
      IDType u = row[i], v = col[i];
      if (!this->directed_ && u > v) {
        std::swap(u, v);
      }
      edges.push_back(std::tuple<IDType, IDType>(u, v));
    }
    sort(edges.begin(), edges.end(),
         [](const std::tuple<IDType, IDType> &t1,
            const std::tuple<IDType, IDType> t2) {
           if (std::get<0>(t1) == std::get<0>(t2)) {
             return std::get<1>(t1) < std::get<1>(t2);
           } else {
             return std::get<0>(t1) < std::get<0>(t2);
           }
         });
    
      auto unique_it = unique(edges.begin(), edges.end(),
                              [](const std::tuple<IDType, IDType> &t1,
                                 const std::tuple<IDType, IDType> t2) {
                                return (std::get<0>(t1) == std::get<0>(t2)) &&
                                       (std::get<1>(t1) == std::get<1>(t2));
                              });
      edges.erase(unique_it, edges.end());
    
    for (int i = 0; i < (int) edges.size(); ++i) {
      edgeListFile << std::get<0>(edges[i]) << " " << std::get<1>(edges[i]) << "\n";
    }
  }
  else {
    IDType* row = coo->get_row();
    IDType* col = coo->get_col();
    ValueType* val = coo->get_vals();
    std::vector<std::tuple<IDType, IDType, ValueType>> edges;
    for (int i = 0; i < coo->get_num_nnz(); ++i) {
        IDType u = row[i], v = col[i];
        ValueType w = 0;
        if (val != nullptr) {
          w = val[i];
        }
        if (!this->directed_ && u > v) {
          std::swap(u, v);
        }
        edges.push_back(std::tuple<IDType, IDType, ValueType>(u, v, w));
      }
      sort(edges.begin(), edges.end(),
           [](const std::tuple<IDType, IDType, ValueType> &t1,
              const std::tuple<IDType, IDType, ValueType> t2) {
             if (std::get<0>(t1) == std::get<0>(t2)) {
               return std::get<1>(t1) < std::get<1>(t2);
             } else {
               return std::get<0>(t1) < std::get<0>(t2);
             }
           });
      
        auto unique_it =
            unique(edges.begin(), edges.end(),
                   [](const std::tuple<IDType, IDType, ValueType> &t1,
                      const std::tuple<IDType, IDType, ValueType> t2) {
                     return (std::get<0>(t1) == std::get<0>(t2)) &&
                            (std::get<1>(t1) == std::get<1>(t2));
                   });
        edges.erase(unique_it, edges.end());
      
      for (int i = 0; i < (int) edges.size(); ++i) {
        edgeListFile << std::get<0>(edges[i]) << " " << std::get<1>(edges[i]);
        if (val != nullptr)
          edgeListFile << " " << std::get<2>(edges[i]);
        edgeListFile << "\n";
      }
    }
    edgeListFile.close();
}

template <typename IDType, typename NNZType, typename ValueType>
void EdgeListWriter<IDType, NNZType, ValueType>::WriteCSR(
    format::CSR<IDType, NNZType, ValueType> *csr) const {
  std::ofstream edgeListFile;
  edgeListFile.open(filename_);
  if constexpr (std::is_same_v<ValueType, void>) {
    NNZType* ind = csr->get_row_ptr();
    IDType* col = csr->get_col();
    std::vector<std::tuple<IDType, IDType>> edges;
    for (int i = 0; i < csr->get_dimensions()[0]; ++i) {
      for (NNZType j = ind[i]; j < ind[i + 1]; ++j) {
        IDType u = i, v = col[j];
        if (!this->directed_ && u > v) {
          std::swap(u, v);
        }
        edges.push_back(std::tuple<IDType, IDType>(u, v));
      }
    }
    sort(edges.begin(), edges.end(),
         [](const std::tuple<IDType, IDType> &t1,
            const std::tuple<IDType, IDType> t2) {
           if (std::get<0>(t1) == std::get<0>(t2)) {
             return std::get<1>(t1) < std::get<1>(t2);
           } else {
             return std::get<0>(t1) < std::get<0>(t2);
           }
         });
      auto unique_it = unique(edges.begin(), edges.end(),
                              [](const std::tuple<IDType, IDType> &t1,
                                 const std::tuple<IDType, IDType> t2) {
                                return (std::get<0>(t1) == std::get<0>(t2)) &&
                                       (std::get<1>(t1) == std::get<1>(t2));
                              });
      edges.erase(unique_it, edges.end());
    
    for (int i = 0; i < (int) edges.size(); ++i) {
      edgeListFile << std::get<0>(edges[i]) << " " << std::get<1>(edges[i]) << "\n";
    }
  }
  else {
    NNZType* ind = csr->get_row_ptr();
    IDType* col = csr->get_col();
    ValueType* val = csr->get_vals();
    std::vector<std::tuple<IDType, IDType, ValueType>> edges;
    for (int i = 0; i < csr->get_dimensions()[0]; ++i) {
      for (NNZType j = ind[i]; j < ind[i + 1]; ++j) {
        IDType u = i, v = col[j];
        ValueType w = 0;
        if (val != nullptr) {
          w = val[j];
        }
        if (!this->directed_ && u > v) {
          std::swap(u, v);
        }
        edges.push_back(std::tuple<IDType, IDType, ValueType>(u, v, w));
      }
    }
    sort(edges.begin(), edges.end(),
         [](const std::tuple<IDType, IDType, ValueType> &t1,
            const std::tuple<IDType, IDType, ValueType> t2) {
           if (std::get<0>(t1) == std::get<0>(t2)) {
             return std::get<1>(t1) < std::get<1>(t2);
           } else {
             return std::get<0>(t1) < std::get<0>(t2);
           }
         });
    
      auto unique_it =
          unique(edges.begin(), edges.end(),
                 [](const std::tuple<IDType, IDType, ValueType> &t1,
                    const std::tuple<IDType, IDType, ValueType> t2) {
                   return (std::get<0>(t1) == std::get<0>(t2)) &&
                          (std::get<1>(t1) == std::get<1>(t2));
                 });
      edges.erase(unique_it, edges.end());
    
    for (int i = 0; i < (int) edges.size(); ++i) {
      edgeListFile << std::get<0>(edges[i]) << " " << std::get<1>(edges[i]);
      if (val != nullptr)
        edgeListFile << " " << std::get<2>(edges[i]);
      edgeListFile << "\n";
    }
  }
  edgeListFile.close();
}

#ifndef _HEADER_ONLY
#include "init/edge_list_writer.inc"
#endif
}  // namespace sparsebase::io

#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/io/edge_list_reader.h"

namespace sparsebase::io{

template <typename IDType, typename NNZType, typename ValueType>
EdgeListReader<IDType, NNZType, ValueType>::EdgeListReader(
    std::string filename, bool weighted, bool remove_duplicates,
    bool remove_self_edges, bool read_undirected, bool square)
    : filename_(filename),
      weighted_(weighted),
      remove_duplicates_(remove_duplicates),
      remove_self_edges_(remove_self_edges),
      read_undirected_(read_undirected),
      square_(square) {}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType>
*EdgeListReader<IDType, NNZType, ValueType>::ReadCOO() const {
  std::ifstream infile(this->filename_);
  if (infile.is_open()) {
    if constexpr (std::is_same_v<ValueType, void>) {
      if (weighted_) {
        throw utils::ReaderException("Cannot read weights into ValueType void");
      }
      IDType u, v;
      IDType m = 0;
      IDType n = 0;
      NNZType nnz = 0;
      std::vector<std::tuple<IDType, IDType>> edges;
      // vertices are 0-based
      while (infile >> u >> v) {
        if (u != v || !remove_self_edges_) {
          edges.push_back(std::tuple<IDType, IDType>(u, v));

          if (read_undirected_)
            edges.push_back(std::tuple<IDType, IDType>(v, u));

          n = std::max(n, u + 1);
          m = std::max(m, v + 1);
        }
      }

      if (square_ || read_undirected_) {
        n = std::max(n, m);
        m = n;
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

      if (remove_duplicates_) {
        auto unique_it = unique(edges.begin(), edges.end(),
                                [](const std::tuple<IDType, IDType> &t1,
                                   const std::tuple<IDType, IDType> t2) {
                                  return (std::get<0>(t1) == std::get<0>(t2)) &&
                                         (std::get<1>(t1) == std::get<1>(t2));
                                });
        edges.erase(unique_it, edges.end());
      }

      nnz = edges.size();

      IDType *row = new IDType[nnz];
      IDType *col = new IDType[nnz];
      ValueType *vals = nullptr;

      for (IDType i = 0; i < nnz; i++) {
        row[i] = std::get<0>(edges[i]);
        col[i] = std::get<1>(edges[i]);
      }

      return new format::COO<IDType, NNZType, ValueType>(n, m, nnz, row, col,
                                                         vals, format::kOwned);
    } else {
      IDType u, v;
      ValueType w = 0;
      IDType m = 0;
      IDType n = 0;
      NNZType nnz = 0;
      std::vector<std::tuple<IDType, IDType, ValueType>> edges;
      // vertices are 0-based
      while (infile >> u >> v) {
        if (weighted_) {
          infile >> w;
        }

        if (u != v || !remove_self_edges_) {
          edges.push_back(std::tuple<IDType, IDType, ValueType>(u, v, w));

          if (read_undirected_)
            edges.push_back(std::tuple<IDType, IDType, ValueType>(v, u, w));

          n = std::max(n, u + 1);
          m = std::max(m, v + 1);
        }
      }

      if (square_ || read_undirected_) {
        n = std::max(n, m);
        m = n;
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

      if (remove_duplicates_) {
        auto unique_it =
            unique(edges.begin(), edges.end(),
                   [](const std::tuple<IDType, IDType, ValueType> &t1,
                      const std::tuple<IDType, IDType, ValueType> t2) {
                     return (std::get<0>(t1) == std::get<0>(t2)) &&
                            (std::get<1>(t1) == std::get<1>(t2));
                   });
        edges.erase(unique_it, edges.end());
      }

      nnz = edges.size();

      IDType *row = new IDType[nnz];
      IDType *col = new IDType[nnz];
      ValueType *vals = nullptr;
      if (weighted_) {
        vals = new ValueType[nnz];
      }

      for (IDType i = 0; i < nnz; i++) {
        row[i] = std::get<0>(edges[i]);
        col[i] = std::get<1>(edges[i]);

        if (weighted_) vals[i] = std::get<2>(edges[i]);
      }

      return new format::COO<IDType, NNZType, ValueType>(n, m, nnz, row, col,
                                                         vals, format::kOwned);
    }

  } else {
    throw utils::ReaderException("file does not exist!");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType>
*EdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  auto coo = ReadCOO();
  utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType> converterObj;
  context::CPUContext cpu_context;
  return converterObj.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, &cpu_context);
}

template <typename IDType, typename NNZType, typename ValueType>
EdgeListReader<IDType, NNZType, ValueType>::~EdgeListReader(){};
#ifndef _HEADER_ONLY
#include "init/edge_list_reader.inc"
#endif
}

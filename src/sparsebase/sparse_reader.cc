#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_exception.h"
#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "sparsebase/sparse_file_format.h"
#ifdef USE_PIGO
#include "external/pigo/pigo.hpp"
#endif

using namespace sparsebase::format;

namespace sparsebase {

namespace utils {

template <typename IDType, typename NNZType, typename ValueType>
Reader<IDType, NNZType, ValueType>::~Reader(){};

// Add weighted option with contexpr
//! Brief description
/*!
  Detailed description
  \param filename string
  \param weighted bool
  \return std::vector of formats
*/
template <typename VertexID, typename NumEdges, typename Weight>
UedgelistReader<VertexID, NumEdges, Weight>::UedgelistReader(
    std::string filename, bool weighted)
    : filename_(filename), weighted_(weighted) {}
template <typename VertexID, typename NumEdges, typename Weight>
Format *UedgelistReader<VertexID, NumEdges, Weight>::ReadSparseFormat() const {
  return this->ReadCSR();
}
template <typename VertexID, typename NumEdges, typename Weight>
CSR<VertexID, NumEdges, Weight> *
UedgelistReader<VertexID, NumEdges, Weight>::ReadCSR() const {
  std::ifstream infile(this->filename_);
  if (infile.is_open()) {
    VertexID u, v;
    NumEdges edges_read = 0;
    VertexID n = 0;

    std::vector<std::pair<VertexID, VertexID>> edges;
    // vertices are 0-based
    while (infile >> u >> v) {
      if (u != v) {
        edges.push_back(std::pair<VertexID, VertexID>(u, v));
        edges.push_back(std::pair<VertexID, VertexID>(v, u));

        n = std::max(n, u);
        n = std::max(n, v);

        edges_read++;
      }
    }
    n++;
    std::cout << "No vertices is " << n << std::endl;
    std::cout << "No read edges " << edges_read << std::endl;
    NumEdges m = edges.size();
    std::cout << "No edges is " << m << std::endl;

    sort(edges.begin(), edges.end(), SortEdge);
    edges.erase(unique(edges.begin(), edges.end()), edges.end());

    // allocate the memory
    NumEdges *row_ptr = new NumEdges[n + 1];
    VertexID *col = new VertexID[m];
    VertexID *tadj = new VertexID[m];
    VertexID *is = new VertexID[m];

    // populate col and row_ptr
    memset(row_ptr, 0, sizeof(NumEdges) * (n + 1));
    int mt = 0;
    for (std::pair<VertexID, VertexID> &e : edges) {
      row_ptr[e.first + 1]++;
      is[mt] = e.first;
      col[mt++] = e.second;
    }

    for (NumEdges i = 1; i <= n; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    for (VertexID i = 0; i < m; i++) {
      tadj[i] = row_ptr[col[i]]++;
    }
    for (NumEdges i = n; i > 0; i--) {
      row_ptr[i] = row_ptr[i - 1];
    }
    row_ptr[0] = 0;
    return new CSR<VertexID, NumEdges, Weight>(n, n, row_ptr, col, nullptr,
                                               kOwned);
  } else {
    throw ReaderException("file does not exists!!");
  }
}
template <typename VertexID, typename NumEdges, typename Weight>
bool UedgelistReader<VertexID, NumEdges, Weight>::SortEdge(
    const std::pair<VertexID, VertexID> &a,
    const std::pair<VertexID, VertexID> &b) {
  if (a.first == b.first) {
    return (a.second < b.second);
  } else {
    return (a.first < b.first);
  }
}
template <typename VertexID, typename NumEdges, typename Weight>
UedgelistReader<VertexID, NumEdges, Weight>::~UedgelistReader(){};
template <typename VertexID, typename NumEdges, typename Weight>
Format *MTXReader<VertexID, NumEdges, Weight>::ReadSparseFormat() const {
  return this->ReadCOO();
}

template <typename VertexID, typename NumEdges, typename Weight>
MTXReader<VertexID, NumEdges, Weight>::MTXReader(std::string filename,
                                                 bool weighted)
    : filename_(filename), weighted_(weighted) {}

template <typename VertexID, typename NumEdges, typename Weight>
COO<VertexID, NumEdges, Weight> *
MTXReader<VertexID, NumEdges, Weight>::ReadCOO() const {
  // Open the file:
  std::ifstream fin(filename_);

  // Declare variables: (check the types here)
  VertexID M, N, L;

  // Ignore headers and comments:
  while (fin.peek() == '%')
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  fin >> M >> N >> L;

  VertexID *row = new VertexID[L];
  VertexID *col = new VertexID[L];
  std::cout << "weighted " << weighted_ << std::endl;
  if (weighted_) {
    if constexpr (!std::is_same_v<void, Weight>) {
      Weight *vals = new Weight[L];
      for (NumEdges l = 0; l < L; l++) {
        VertexID m, n;
        Weight w;
        fin >> m >> n >> w;
        row[l] = n - 1;
        col[l] = m - 1;
        vals[l] = w;
      }

      auto coo =
          new COO<VertexID, NumEdges, Weight>(M, N, L, row, col, vals, kOwned);
      return coo;
    } else {
      // TODO: Add an exception class for this
      throw ReaderException("Weight type for weighted graphs can not be void");
    }
  } else {
    for (NumEdges l = 0; l < L; l++) {
      VertexID m, n;
      fin >> m >> n;
      row[l] = m - 1;
      col[l] = n - 1;
    }

    auto coo =
        new COO<VertexID, NumEdges, Weight>(M, N, L, row, col, nullptr, kOwned);
    return coo;
  }
}

template <typename VertexID, typename NumEdges, typename Weight>
MTXReader<VertexID, NumEdges, Weight>::~MTXReader(){};


#ifdef USE_PIGO
template <typename IDType, typename NNZType, typename ValueType>
PigoMTXReader<IDType, NNZType, ValueType>::PigoMTXReader
    (std::string filename, bool weighted, bool _convert_to_zero_index)
    : filename_(filename), weighted_(weighted), convert_to_zero_index_(_convert_to_zero_index) {}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType> *
PigoMTXReader<IDType, NNZType, ValueType>::ReadCOO() const {

  COO<IDType, NNZType, ValueType> * coo;

  if(weighted_){
    pigo::COO<IDType, IDType, IDType*, false, false, false, true, ValueType, ValueType*>
        pigo_coo(filename_, pigo::MATRIX_MARKET);
    coo = new COO<IDType, NNZType, ValueType>(pigo_coo.n(), pigo_coo.m(),
                                              pigo_coo.m(), pigo_coo.x(),
                                              pigo_coo.y(), pigo_coo.w(), kOwned);
  }
  else {
    pigo::COO<IDType, IDType, IDType*, false, false, false, false, ValueType, ValueType*>
        pigo_coo(filename_, pigo::MATRIX_MARKET);
    coo = new COO<IDType, NNZType, ValueType>(pigo_coo.n(), pigo_coo.m(),
                                              pigo_coo.m(), pigo_coo.x(),
                                              pigo_coo.y(), pigo_coo.w(), kOwned);
  }

  if(convert_to_zero_index_){
    auto col = coo->get_col();
    auto row = coo->get_row();
#pragma omp parallel for shared(col,row,coo)
    for (IDType i = 0; i < coo->get_num_nnz(); ++i) {
      col[i]--;
      row[i]--;
    }
  }

  return coo;
}


template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> *
PigoMTXReader<IDType, NNZType, ValueType>::ReadCSR() const {
  COO<IDType, NNZType, ValueType>* coo = ReadCOO();
  utils::Converter<IDType,NNZType,ValueType> converter;
  return converter.template Convert<CSR<IDType,NNZType,ValueType>>(coo, coo->get_context(), true);
}

template <typename IDType, typename NNZType, typename ValueType>
Format *PigoMTXReader<IDType, NNZType, ValueType>::ReadSparseFormat() const {
  return this->ReadCOO();
}


template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> *
PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  COO<IDType, NNZType, ValueType>* coo = ReadCOO();
  utils::Converter<IDType,NNZType,ValueType> converter;
  return converter.template Convert<CSR<IDType,NNZType,ValueType>>(coo, coo->get_context(), true);
}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType> *
PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCOO() const {
  if(weighted_){
    pigo::COO<IDType, IDType, IDType*, false, false, false, true, ValueType, ValueType*> coo(filename_, pigo::EDGE_LIST);
    return new COO<IDType, NNZType, ValueType>(coo.n(), coo.m(), coo.m(), coo.x(), coo.y(), coo.w(), kOwned);
  }
  else {
    pigo::COO<IDType, IDType, IDType*, false, false, false, false, ValueType, ValueType*> coo(filename_, pigo::EDGE_LIST);
    return new COO<IDType, NNZType, ValueType>(coo.n(), coo.m(), coo.m(), coo.x(), coo.y(), coo.w(), kOwned);
  }
}

template <typename IDType, typename NNZType, typename ValueType>
PigoEdgeListReader<IDType, NNZType, ValueType>::PigoEdgeListReader(
    std::string filename, bool weighted)
    : filename_(filename), weighted_(weighted) {}


template <typename IDType, typename NNZType, typename ValueType>
Format *PigoEdgeListReader<IDType, NNZType, ValueType>::ReadSparseFormat() const {
  return this->ReadCOO();
}
#if !defined(_HEADER_ONLY)
#include "init/external/pigo.inc"
#endif
#endif


template <typename IDType, typename NNZType, typename ValueType>
BinaryReader<IDType, NNZType, ValueType>::BinaryReader(std::string filename) : filename_(filename) {}



template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> *
BinaryReader<IDType, NNZType, ValueType>::ReadCSR() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if(sbff.get_name() != "csr"){
    throw utils::ReaderException("SBFF file is not in CSR format");
  }

  NNZType* row_ptr;
  IDType* col;
  ValueType* vals = nullptr;

  auto dimensions = sbff.get_dimensions();

  sbff.template GetArray("row_ptr", row_ptr);
  sbff.template GetArray("col", col);

  if(sbff.get_array_count() == 3){
    sbff.template GetArray("vals", vals);
  }

  return new CSR<IDType, NNZType, ValueType>(dimensions[0], dimensions[1], row_ptr, col, vals, kOwned);
}


template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType> *
BinaryReader<IDType, NNZType, ValueType>::ReadCOO() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if(sbff.get_name() != "coo"){
    throw utils::ReaderException("SBFF file is not in COO format");
  }

  IDType* row;
  IDType* col;
  ValueType* vals = nullptr;

  auto dimensions = sbff.get_dimensions();

  sbff.template GetArray("row", row);
  sbff.template GetArray("col", col);

  if(sbff.get_array_count() == 3){
    sbff.template GetArray("vals", vals);
  }

  return new COO<IDType, NNZType, ValueType>(dimensions[0], dimensions[1], dimensions[1], row, col, vals, kOwned);
}

#if !defined(_HEADER_ONLY)
#include "init/reader.inc"
#endif

} // namespace utils

} // namespace sparsebase

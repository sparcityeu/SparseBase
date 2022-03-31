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
UedgelistReader<IDType, NNZType, ValueType>::UedgelistReader(
    std::string filename, bool weighted)
    : filename_(filename), weighted_(weighted) {}


template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> *
UedgelistReader<IDType, NNZType, ValueType>::ReadCSR() const {
  std::ifstream infile(this->filename_);
  if (infile.is_open()) {
    IDType u, v;
    NNZType edges_read = 0;
    IDType n = 0;

    std::vector<std::pair<IDType, IDType>> edges;
    // vertices are 0-based
    while (infile >> u >> v) {
      if (u != v) {
        edges.push_back(std::pair<IDType, IDType>(u, v));
        edges.push_back(std::pair<IDType, IDType>(v, u));

        n = std::max(n, u);
        n = std::max(n, v);

        edges_read++;
      }
    }
    n++;
    std::cout << "No vertices is " << n << std::endl;
    std::cout << "No read edges " << edges_read << std::endl;
    NNZType m = edges.size();
    std::cout << "No edges is " << m << std::endl;

    sort(edges.begin(), edges.end(), SortEdge);
    edges.erase(unique(edges.begin(), edges.end()), edges.end());

    // allocate the memory
    NNZType *row_ptr = new NNZType[n + 1];
    IDType *col = new IDType[m];
    IDType *tadj = new IDType[m];
    IDType *is = new IDType[m];

    // populate col and row_ptr
    memset(row_ptr, 0, sizeof(NNZType) * (n + 1));
    int mt = 0;
    for (std::pair<IDType, IDType> &e : edges) {
      row_ptr[e.first + 1]++;
      is[mt] = e.first;
      col[mt++] = e.second;
    }

    for (NNZType i = 1; i <= n; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    for (IDType i = 0; i < m; i++) {
      tadj[i] = row_ptr[col[i]]++;
    }
    for (NNZType i = n; i > 0; i--) {
      row_ptr[i] = row_ptr[i - 1];
    }
    row_ptr[0] = 0;
    return new CSR<IDType, NNZType, ValueType>(n, n, row_ptr, col, nullptr,
                                               kOwned);
  } else {
    throw ReaderException("file does not exists!!");
  }
}
template <typename IDType, typename NNZType, typename ValueType>
bool UedgelistReader<IDType, NNZType, ValueType>::SortEdge(
    const std::pair<IDType, IDType> &a,
    const std::pair<IDType, IDType> &b) {
  if (a.first == b.first) {
    return (a.second < b.second);
  } else {
    return (a.first < b.first);
  }
}
template <typename IDType, typename NNZType, typename ValueType>
UedgelistReader<IDType, NNZType, ValueType>::~UedgelistReader(){};


template <typename IDType, typename NNZType, typename ValueType>
MTXReader<IDType, NNZType, ValueType>::MTXReader(std::string filename,
                                                 bool weighted)
    : filename_(filename), weighted_(weighted) {}

template <typename IDType, typename NNZType, typename ValueType>
COO<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCOO() const {
  // Open the file:
  std::ifstream fin(filename_);

  if(fin.is_open()){
    // Declare variables: (check the types here)
    DimensionType M, N, L;

    // Ignore headers and comments:
    while (fin.peek() == '%')
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    fin >> M >> N >> L;

    IDType *row = new IDType[L];
    IDType *col = new IDType[L];
    if (weighted_) {
      if constexpr (!std::is_same_v<void, ValueType>) {
        ValueType *vals = new ValueType[L];
        for (NNZType l = 0; l < L; l++) {
          IDType m, n;
          ValueType w;
          fin >> m >> n >> w;
          row[l] = n - 1;
          col[l] = m - 1;
          vals[l] = w;
        }

        auto coo =
            new COO<IDType, NNZType, ValueType>(M, N, L, row, col, vals, kOwned);
        return coo;
      } else {
        // TODO: Add an exception class for this
        throw ReaderException("Weight type for weighted graphs can not be void");
      }
    } else {
      for (NNZType l = 0; l < L; l++) {
        IDType m, n;
        fin >> m >> n;
        row[l] = m - 1;
        col[l] = n - 1;
      }

      auto coo =
          new COO<IDType, NNZType, ValueType>(M, N, L, row, col, nullptr, kOwned);
      return coo;
    }
  } else {
    throw ReaderException("file does not exists!!");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
MTXReader<IDType, NNZType, ValueType>::~MTXReader(){};


#ifdef USE_PIGO
template <typename IDType, typename NNZType, typename ValueType>
PigoMTXReader<IDType, NNZType, ValueType>::PigoMTXReader
    (std::string filename, bool weighted, bool convert_to_zero_index)
    : filename_(filename), weighted_(weighted), convert_to_zero_index_(convert_to_zero_index) {}

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
  utils::OrderTwoConverter<IDType,NNZType,ValueType> converter;
  return converter.template Convert<CSR<IDType,NNZType,ValueType>>(coo, coo->get_context(), true);
}


template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> *
PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  COO<IDType, NNZType, ValueType>* coo = ReadCOO();
  utils::OrderTwoConverter<IDType,NNZType,ValueType> converter;
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



#if !defined(_HEADER_ONLY)
#include "init/external/pigo.inc"
#endif
#endif


template <typename IDType, typename NNZType, typename ValueType>
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::BinaryReaderOrderTwo(std::string filename) : filename_(filename) {}



template <typename IDType, typename NNZType, typename ValueType>
CSR<IDType, NNZType, ValueType> *
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::ReadCSR() const {
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
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::ReadCOO() const {
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

template <typename T>
BinaryReaderOrderOne<T>::BinaryReaderOrderOne(std::string filename) : filename_(filename) {}

template <typename T> Array<T> *BinaryReaderOrderOne<T>::ReadArray() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if(sbff.get_name() != "array"){
    throw utils::ReaderException("SBFF file is not in Array format");
  }

  DimensionType size = sbff.get_dimensions()[0];
  T* arr;
  sbff.template GetArray("array", arr);

  return new Array<T>(size, arr, kOwned);
}

#if !defined(_HEADER_ONLY)
#include "init/reader.inc"
#endif

} // namespace utils

} // namespace sparsebase

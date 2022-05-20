#include "sparsebase/utils/io/reader.h"
#include "sparsebase/format/format.h"
#include "sparsebase/utils/converter/converter.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/utils/io/sparse_file_format.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#ifdef USE_PIGO
#include "sparsebase/external/pigo/pigo.hpp"
#endif

namespace sparsebase {

namespace utils {

namespace io {

template <typename IDType, typename NNZType, typename ValueType>
EdgeListReader<IDType, NNZType, ValueType>::EdgeListReader(
    std::string filename, bool weighted, bool remove_duplicates, bool remove_self_edges, bool read_undirected, bool square)
    : filename_(filename), weighted_(weighted),
      remove_duplicates_(remove_duplicates),
      remove_self_edges_(remove_self_edges),
      read_undirected_(read_undirected),
      square_(square)
{}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType> *
EdgeListReader<IDType, NNZType, ValueType>::ReadCOO() const {
  std::ifstream infile(this->filename_);
  if (infile.is_open()) {
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

        if(read_undirected_)
          edges.push_back(std::tuple<IDType, IDType, ValueType>(v, u, w));

        n = std::max(n, u+1);
        m = std::max(m, v+1);

      }

    }

    if(square_){
      n = std::max(n,m);
      m = n;
    }

    sort(edges.begin(), edges.end(),
         [](const std::tuple<IDType, IDType, ValueType>& t1,
            const std::tuple<IDType, IDType, ValueType> t2){
      if(std::get<0>(t1) == std::get<0>(t2)){
        return std::get<1>(t1) < std::get<1>(t2);
      } else {
        return std::get<0>(t1) < std::get<0>(t2);
      }
    });

    if(remove_duplicates_){
      auto unique_it = unique(edges.begin(), edges.end(),
                              [](const std::tuple<IDType, IDType, ValueType>& t1,
                                 const std::tuple<IDType, IDType, ValueType> t2){
        return (std::get<0>(t1) == std::get<0>(t2)) && (std::get<1>(t1) == std::get<1>(t2));
      });
      edges.erase(unique_it, edges.end());
    }

    nnz = edges.size();

    IDType* row = new IDType[nnz];
    IDType* col = new IDType[nnz];
    ValueType* vals = nullptr;
    if(weighted_){
      vals = new ValueType[nnz];
    }

    for(IDType i=0; i<nnz; i++){
      row[i] = std::get<0>(edges[i]);
      col[i] = std::get<1>(edges[i]);

      if(weighted_)
        vals[i] = std::get<2>(edges[i]);
    }

    return new format::COO<IDType, NNZType, ValueType>(n,m,nnz,row,col,vals,format::kOwned);

  } else {
    throw ReaderException("file does not exist!");
  }

}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
EdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  auto coo = ReadCOO();
  converter::ConverterOrderTwo<IDType,NNZType, ValueType> converterObj;
  context::CPUContext cpu_context;
  return converterObj.template Convert<format::CSR<IDType, NNZType, ValueType>>(coo, &cpu_context);
}


template <typename IDType, typename NNZType, typename ValueType>
EdgeListReader<IDType, NNZType, ValueType>::~EdgeListReader(){};

template <typename IDType, typename NNZType, typename ValueType>
MTXReader<IDType, NNZType, ValueType>::MTXReader(std::string filename,
                                                 bool weighted,
                                                 bool convert_to_zero_index)
    : filename_(filename), weighted_(weighted),
      convert_to_zero_index_(convert_to_zero_index){}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCOO() const {
  // Open the file:
  std::ifstream fin(filename_);

  if (fin.is_open()) {
    // Declare variables: (check the types here)
    format::DimensionType M, N, L;

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

          if(convert_to_zero_index_){
            n--;
            m--;
          }

          row[l] = m;
          col[l] = n;
          vals[l] = w;
        }

        auto coo = new format::COO<IDType, NNZType, ValueType>(
            M, N, L, row, col, vals, format::kOwned);
        return coo;
      } else {
        // TODO: Add an exception class for this
        throw ReaderException(
            "Weight type for weighted graphs can not be void");
      }
    } else {
      for (NNZType l = 0; l < L; l++) {
        IDType m, n;
        fin >> m >> n;

        if(convert_to_zero_index_){
          n--;
          m--;
        }

        row[l] = m;
        col[l] = n;
      }

      auto coo = new format::COO<IDType, NNZType, ValueType>(
          M, N, L, row, col, nullptr, format::kOwned);
      return coo;
    }
  } else {
    throw ReaderException("file does not exists!!");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCSR() const {
  auto coo = ReadCOO();
  converter::ConverterOrderTwo<IDType,NNZType, ValueType> converterObj;
  context::CPUContext cpu_context;
  return converterObj.template Convert<format::CSR<IDType, NNZType, ValueType>>(coo, &cpu_context);
}


template <typename IDType, typename NNZType, typename ValueType>
MTXReader<IDType, NNZType, ValueType>::~MTXReader(){};


template <typename IDType, typename NNZType, typename ValueType>
PigoMTXReader<IDType, NNZType, ValueType>::PigoMTXReader(
    std::string filename, bool weighted, bool convert_to_zero_index)
    : filename_(filename), weighted_(weighted),
      convert_to_zero_index_(convert_to_zero_index) {}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType> *
PigoMTXReader<IDType, NNZType, ValueType>::ReadCOO() const {

#ifdef USE_PIGO
  format::COO<IDType, NNZType, ValueType> *coo;

  if (weighted_) {
    pigo::COO<IDType, IDType, IDType *, false, false, false, true, ValueType,
              ValueType *>
        pigo_coo(filename_, pigo::MATRIX_MARKET);
    coo = new format::COO<IDType, NNZType, ValueType>(
        pigo_coo.nrows()-1, pigo_coo.ncols()-1, pigo_coo.m(), pigo_coo.x(), pigo_coo.y(),
        pigo_coo.w(), format::kOwned);
  } else {
    pigo::COO<IDType, IDType, IDType *, false, false, false, false, ValueType,
              ValueType *>
        pigo_coo(filename_, pigo::MATRIX_MARKET);
    coo = new format::COO<IDType, NNZType, ValueType>(
        pigo_coo.nrows()-1, pigo_coo.ncols()-1, pigo_coo.m(), pigo_coo.x(), pigo_coo.y(),
        pigo_coo.w(), format::kOwned);
  }

  if (convert_to_zero_index_) {
    auto col = coo->get_col();
    auto row = coo->get_row();
#pragma omp parallel for shared(col, row, coo)
    for (IDType i = 0; i < coo->get_num_nnz(); ++i) {
      col[i]--;
      row[i]--;
    }
  }

  return coo;
#else

  std::cerr << "Warning: PIGO suppport is not compiled in this build of sparsebase (your system might not be supported)." << std::endl;
  std::cerr << "Defaulting to sequential reader" << std::endl;
  MTXReader<IDType, NNZType, ValueType> reader(filename_, weighted_);
  return reader.ReadCOO();
#endif
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
PigoMTXReader<IDType, NNZType, ValueType>::ReadCSR() const {
  format::COO<IDType, NNZType, ValueType> *coo = ReadCOO();
  utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType> converter;
  return converter.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, coo->get_context(), true);
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  format::COO<IDType, NNZType, ValueType> *coo = ReadCOO();
  utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType> converter;
  return converter.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, coo->get_context(), true);
}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType> *
PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCOO() const {
#ifdef USE_PIGO
  if (weighted_) {
    pigo::COO<IDType, IDType, IDType *, false, false, false, true, ValueType,
              ValueType *>
        coo(filename_, pigo::EDGE_LIST);
    return new format::COO<IDType, NNZType, ValueType>(
        coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), coo.w(), format::kOwned);
  } else {
    pigo::COO<IDType, IDType, IDType *, false, false, false, false, ValueType,
              ValueType *>
        coo(filename_, pigo::EDGE_LIST);
    return new format::COO<IDType, NNZType, ValueType>(
        coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), coo.w(), format::kOwned);
  }
#else
  std::cerr << "Warning: PIGO suppport is not compiled in this build of sparsebase (your system might not be supported)." << std::endl;
  std::cerr << "Defaulting to sequential reader" << std::endl;
  EdgeListReader<IDType, NNZType, ValueType> reader(filename_, weighted_, true, true, false);
  return reader.ReadCOO();
#endif
}

template <typename IDType, typename NNZType, typename ValueType>
PigoEdgeListReader<IDType, NNZType, ValueType>::PigoEdgeListReader(
    std::string filename, bool weighted)
    : filename_(filename), weighted_(weighted) {}




template <typename IDType, typename NNZType, typename ValueType>
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::BinaryReaderOrderTwo(
    std::string filename)
    : filename_(filename) {}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::ReadCSR() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if (sbff.get_name() != "csr") {
    throw utils::ReaderException("SBFF file is not in CSR format");
  }

  NNZType *row_ptr;
  IDType *col;
  ValueType *vals = nullptr;

  auto dimensions = sbff.get_dimensions();

  sbff.template GetArray("row_ptr", row_ptr);
  sbff.template GetArray("col", col);

  if (sbff.get_array_count() == 3) {
    sbff.template GetArray("vals", vals);
  }

  return new format::CSR<IDType, NNZType, ValueType>(
      dimensions[0], dimensions[1], row_ptr, col, vals, format::kOwned);
}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType> *
BinaryReaderOrderTwo<IDType, NNZType, ValueType>::ReadCOO() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if (sbff.get_name() != "coo") {
    throw utils::ReaderException("SBFF file is not in COO format");
  }

  IDType *row;
  IDType *col;
  ValueType *vals = nullptr;

  auto dimensions = sbff.get_dimensions();

  sbff.template GetArray("row", row);
  sbff.template GetArray("col", col);

  if (sbff.get_array_count() == 3) {
    sbff.template GetArray("vals", vals);
  }

  return new format::COO<IDType, NNZType, ValueType>(
      dimensions[0], dimensions[1], dimensions[1], row, col, vals,
      format::kOwned);
}

template <typename T>
BinaryReaderOrderOne<T>::BinaryReaderOrderOne(std::string filename)
    : filename_(filename) {}

template <typename T>
format::Array<T> *BinaryReaderOrderOne<T>::ReadArray() const {
  auto sbff = SbffObject::ReadObject(filename_);

  if (sbff.get_name() != "array") {
    throw utils::ReaderException("SBFF file is not in Array format");
  }

  format::DimensionType size = sbff.get_dimensions()[0];
  T *arr;
  sbff.template GetArray("array", arr);

  return new format::Array<T>(size, arr, format::kOwned);
}

#if !defined(_HEADER_ONLY)
#include "init/reader.inc"
#include "init/external/pigo.inc"
#endif


} // namespace io

} // namespace utils

} // namespace sparsebase
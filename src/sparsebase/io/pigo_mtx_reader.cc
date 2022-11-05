#include "sparsebase/config.h"
#include "pigo_mtx_reader.h"
#include <string>
#include <fstream>
#ifdef USE_PIGO
#include "sparsebase/external/pigo/pigo.hpp"
#endif
namespace sparsebase::io{
template <typename IDType, typename NNZType, typename ValueType>
PigoMTXReader<IDType, NNZType, ValueType>::PigoMTXReader(
    std::string filename, bool weighted, bool convert_to_zero_index)
    : filename_(filename),
      weighted_(weighted),
      convert_to_zero_index_(convert_to_zero_index) {}

// template <typename IDType, typename NNZType, typename ValueType>
// format::Array<ValueType> *
// PigoMTXReader<IDType, NNZType, ValueType>::ReadArray() const {
//  MTXReader<IDType, NNZType, ValueType> reader(filename_, weighted_);
//  return reader.ReadArray();
//}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType>
*PigoMTXReader<IDType, NNZType, ValueType>::ReadCOO() const {
#ifdef USE_PIGO
  format::COO<IDType, NNZType, ValueType> *coo;

  if (weighted_) {
    if constexpr (!std::is_same_v<ValueType, void>) {
      pigo::COO<IDType, IDType, IDType *, false, false, false, true, ValueType,
                ValueType *>
          pigo_coo(filename_, pigo::MATRIX_MARKET);
      coo = new format::COO<IDType, NNZType, ValueType>(
          pigo_coo.nrows() - 1, pigo_coo.ncols() - 1, pigo_coo.m(),
          pigo_coo.x(), pigo_coo.y(), pigo_coo.w(), format::kOwned);
    } else {
      throw utils::ReaderException(
          "Cannot read a matrix market with weights into Format with void "
          "ValueType");
    }
  } else {
    if constexpr (!std::is_same_v<ValueType, void>) {
      pigo::COO<IDType, IDType, IDType *, false, false, false, false, ValueType,
                ValueType *>
          pigo_coo(filename_, pigo::MATRIX_MARKET);
      coo = new format::COO<IDType, NNZType, ValueType>(
          pigo_coo.nrows() - 1, pigo_coo.ncols() - 1, pigo_coo.m(),
          pigo_coo.x(), pigo_coo.y(), nullptr, format::kOwned);
    } else {
      pigo::COO<IDType, IDType, IDType *, false, false, false, false, char,
                char *>
          pigo_coo(filename_, pigo::MATRIX_MARKET);
      coo = new format::COO<IDType, NNZType, ValueType>(
          pigo_coo.nrows() - 1, pigo_coo.ncols() - 1, pigo_coo.m(),
          pigo_coo.x(), pigo_coo.y(), nullptr, format::kOwned);
    }
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

  std::cerr << "Warning: PIGO suppport is not compiled in this build of "
               "sparsebase (your system might not be supported)."
            << std::endl;
  std::cerr << "Defaulting to sequential reader" << std::endl;
  MTXReader<IDType, NNZType, ValueType> reader(filename_);
  return reader.ReadCOO();
#endif
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType>
*PigoMTXReader<IDType, NNZType, ValueType>::ReadCSR() const {
  format::COO<IDType, NNZType, ValueType> *coo = ReadCOO();
  converter::ConverterOrderTwo<IDType, NNZType, ValueType> converter;
  std::cout << "nnz " << coo->get_num_nnz() << " dim "
            << coo->get_dimensions()[0] << " " << coo->get_dimensions()[1]
            << std::endl;
  return converter.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, coo->get_context(), true);
}
#ifndef _HEADER_ONLY
#include "init/pigo_mtx_reader.inc"
#endif
}
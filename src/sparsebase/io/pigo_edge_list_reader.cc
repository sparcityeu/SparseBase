#include "sparsebase/config.h"
#include "pigo_edge_list_reader.h"
#include <string>
#include <fstream>
#ifdef USE_PIGO
#include "sparsebase/external/pigo/pigo.hpp"
#endif
namespace sparsebase::io{

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType>
*PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  format::COO<IDType, NNZType, ValueType> *coo = ReadCOO();
  converter::ConverterOrderTwo<IDType, NNZType, ValueType> converter;
  return converter.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, coo->get_context(), true);
}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType>
*PigoEdgeListReader<IDType, NNZType, ValueType>::ReadCOO() const {
#ifdef USE_PIGO
  if (weighted_) {
    if constexpr (!std::is_same_v<ValueType, void>) {
      pigo::COO<IDType, IDType, IDType *, false, false, false, true, ValueType,
                ValueType *>
          coo(filename_, pigo::EDGE_LIST);
      return new format::COO<IDType, NNZType, ValueType>(
          coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), coo.w(),
          format::kOwned);
    } else {
      throw utils::ReaderException(
          "Cannot read a weighted edge list into format with void ValueType");
    }
  } else {
    if constexpr (!std::is_same_v<ValueType, void>) {
      pigo::COO<IDType, IDType, IDType *, false, false, false, false, ValueType,
                ValueType *>
          coo(filename_, pigo::EDGE_LIST);
      return new format::COO<IDType, NNZType, ValueType>(
          coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), nullptr,
          format::kOwned);
    } else {
      pigo::COO<IDType, IDType, IDType *, false, false, false, false, char,
                char *>
          coo(filename_, pigo::EDGE_LIST);
      return new format::COO<IDType, NNZType, ValueType>(
          coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), nullptr,
          format::kOwned);
    }
  }
#else
  std::cerr << "Warning: PIGO suppport is not compiled in this build of "
               "sparsebase (your system might not be supported)."
            << std::endl;
  std::cerr << "Defaulting to sequential reader" << std::endl;
  EdgeListReader<IDType, NNZType, ValueType> reader(filename_, weighted_, true,
                                                    true, false);
  return reader.ReadCOO();
#endif
}

template <typename IDType, typename NNZType, typename ValueType>
PigoEdgeListReader<IDType, NNZType, ValueType>::PigoEdgeListReader(
    std::string filename, bool weighted)
    : filename_(filename), weighted_(weighted) {}
#ifndef _HEADER_ONLY
#include "init/pigo_edge_list_reader.inc"
#endif
}

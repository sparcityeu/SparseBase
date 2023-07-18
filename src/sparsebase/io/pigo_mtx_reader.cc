#include "pigo_mtx_reader.h"

#include <fstream>
#include <sstream>
#include <string>
#include "sparsebase/utils/logger.h"
#include "sparsebase/config.h"
#include "sparsebase/io/mtx_reader.h"
#ifdef USE_PIGO
#include "sparsebase/external/pigo/pigo.hpp"
#endif
namespace sparsebase::io {
template <typename IDType, typename NNZType, typename ValueType>
PigoMTXReader<IDType, NNZType, ValueType>::PigoMTXReader(
    std::string filename, bool weighted, bool convert_to_zero_index)
    : filename_(filename),
      weighted_(weighted),
      convert_to_zero_index_(convert_to_zero_index) {
  std::ifstream fin(filename_);

  if (fin.is_open()) {
    std::string header_line;
    std::getline(fin, header_line);
    // parse first line
    options_ = ParseHeader(header_line);
  } else {
    throw utils::ReaderException("Wrong matrix market file name\n");
  }

  }

// template <typename IDType, typename NNZType, typename ValueType>
// format::Array<ValueType> *
// PigoMTXReader<IDType, NNZType, ValueType>::ReadArray() const {
//  MTXReader<IDType, NNZType, ValueType> reader(filename_, weighted_);
//  return reader.ReadArray();
//}

template <typename IDType, typename NNZType, typename ValueType>
typename PigoMTXReader<IDType, NNZType, ValueType>::MTXOptions
PigoMTXReader<IDType, NNZType, ValueType>::ParseHeader(
    std::string header_line) const {
  std::stringstream line_ss(header_line);
  MTXOptions options;
  std::string prefix, object, format, field, symmetry;
  line_ss >> prefix >> object >> format >> field >> symmetry;
  if (prefix != MMX_PREFIX)
    throw utils::ReaderException("Wrong prefix in a matrix market file");
  // parsing Object option
  if (object == "matrix") {
    options.object =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXObjectOptions::matrix;
  } else if (object == "vector") {
    options.object =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXObjectOptions::matrix;
    throw utils::ReaderException(
        "Matrix market reader does not currently support reading vectors.");
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'object' option in matrix market header");
  }
  // parsing format option
  if (format == "array") {
    options.format =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFormatOptions::array;
  } else if (format == "coordinate") {
    options.format =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFormatOptions::coordinate;
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'format' option in matrix market header");
  }
  // parsing field option
  if (field == "real") {
    options.field =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::real;
    if constexpr (std::is_same<void, ValueType>::value)
      throw utils::ReaderException(
          "You are reading the values of the matrix market file into a void "
          "array");
  } else if (field == "double") {
    options.field =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::double_field;
    if constexpr (std::is_same<void, ValueType>::value)
      throw utils::ReaderException(
          "You are reading the values of the matrix market file into a void "
          "array");
  } else if (field == "complex") {
    options.field =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::complex;
    if constexpr (std::is_same<void, ValueType>::value)
      throw utils::ReaderException(
          "You are reading the values of the matrix market file into a void "
          "array");
  } else if (field == "integer") {
    options.field =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::integer;
    if constexpr (std::is_same<void, ValueType>::value)
      throw utils::ReaderException(
          "You are reading the values of the matrix market file into a void "
          "array");
  } else if (field == "pattern") {
    options.field =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::pattern;
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'field' option in matrix market header");
  }
  // parsing symmetry
  if (symmetry == "general") {
    options.symmetry =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXSymmetryOptions::general;
  } else if (symmetry == "symmetric") {
    options.symmetry =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXSymmetryOptions::symmetric;
  } else if (symmetry == "skew-symmetric") {
    options.symmetry = PigoMTXReader<IDType, NNZType,
                                 ValueType>::MTXSymmetryOptions::skew_symmetric;
  } else if (symmetry == "hermitian") {
    options.symmetry =
        PigoMTXReader<IDType, NNZType, ValueType>::MTXSymmetryOptions::hermitian;
    throw utils::ReaderException(
        "Matrix market reader does not currently support hermitian symmetry.");
  } else {
    throw utils::ReaderException(
        "Illegal value for the 'symmetry' option in matrix market header");
  }
  return options;
}

template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType>
    *PigoMTXReader<IDType, NNZType, ValueType>::ReadCOO() const {
#ifdef USE_PIGO

  format::COO<IDType, NNZType, ValueType> *coo;

  if (options_.object != matrix ||
   options_.format != coordinate ||
   options_.symmetry != general)
   {
      utils::Logger logger(typeid(this));
      logger.Log("PIGO is not equipped to read these options, defaulting to sequential reader...", utils::LOG_LVL_WARNING);
      
      sparsebase::io::MTXReader<IDType, NNZType, ValueType> reader(filename_);
      coo = reader.ReadCOO();
      return coo;
   }

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
}  // namespace sparsebase::io
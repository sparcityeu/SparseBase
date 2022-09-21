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

#define MMX_PREFIX "%%MatrixMarket"
template <typename IDType, typename NNZType, typename ValueType>
EdgeListReader<IDType, NNZType, ValueType>::EdgeListReader(
    std::string filename, bool weighted, bool remove_duplicates,
    bool remove_self_edges, bool read_undirected, bool square)
    : filename_(filename), weighted_(weighted),
      remove_duplicates_(remove_duplicates),
      remove_self_edges_(remove_self_edges), read_undirected_(read_undirected),
      square_(square) {}

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

      if (weighted_)
        vals[i] = std::get<2>(edges[i]);
    }

    return new format::COO<IDType, NNZType, ValueType>(n, m, nnz, row, col,
                                                       vals, format::kOwned);

  } else {
    throw ReaderException("file does not exist!");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
EdgeListReader<IDType, NNZType, ValueType>::ReadCSR() const {
  auto coo = ReadCOO();
  converter::ConverterOrderTwo<IDType, NNZType, ValueType> converterObj;
  context::CPUContext cpu_context;
  return converterObj.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, &cpu_context);
}

template <typename IDType, typename NNZType, typename ValueType>
EdgeListReader<IDType, NNZType, ValueType>::~EdgeListReader(){};

template <typename IDType, typename NNZType, typename ValueType>
MTXReader<IDType, NNZType, ValueType>::MTXReader(std::string filename,
                                                 bool convert_to_zero_index)
    : filename_(filename),
      convert_to_zero_index_(convert_to_zero_index) {
  std::ifstream fin(filename_);

  if (fin.is_open()) {
    std::string header_line;
    std::getline(fin, header_line);
    // parse first line
    options_ = ParseHeader(header_line);
  } else {
    throw ReaderException("Wrong matrix market file name\n");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
typename MTXReader<IDType, NNZType, ValueType>::MTXOptions
MTXReader<IDType, NNZType, ValueType>::ParseHeader(
    std::string header_line) const {
  std::stringstream line_ss(header_line);
  MTXOptions options;
  std::string prefix, object, format, field, symmetry;
  line_ss >> prefix >> object >> format >> field >> symmetry;
  if (prefix != MMX_PREFIX)
    throw ReaderException("Wrong prefix in a matrix market file");
  // parsing Object option
  if (object == "matrix") {
    options.object =
        MTXReader<IDType, NNZType, ValueType>::MTXObjectOptions::matrix;
  } else if (object == "vector") {
    options.object =
        MTXReader<IDType, NNZType, ValueType>::MTXObjectOptions::matrix;
    throw ReaderException("Matrix market reader does not currently support reading vectors.");
  } else {
    throw ReaderException(
        "Illegal value for the 'object' option in matrix market header");
  }
  // parsing format option
  if (format == "array") {
    options.format =
        MTXReader<IDType, NNZType, ValueType>::MTXFormatOptions::array;
  } else if (format == "coordinate") {
    options.format =
        MTXReader<IDType, NNZType, ValueType>::MTXFormatOptions::coordinate;
  } else {
    throw ReaderException(
        "Illegal value for the 'format' option in matrix market header");
  }
  // parsing field option
  if (field == "real") {
    options.field =
        MTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::real;
    if constexpr(std::is_same<void, ValueType>::value) throw ReaderException("You are reading the values of the matrix market file into a void array");
  } else if (field == "double") {
    options.field =
        MTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::double_field;
    if constexpr(std::is_same<void, ValueType>::value) throw ReaderException("You are reading the values of the matrix market file into a void array");
  } else if (field == "complex") {
    options.field =
        MTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::complex;
    if constexpr(std::is_same<void, ValueType>::value) throw ReaderException("You are reading the values of the matrix market file into a void array");
  } else if (field == "integer") {
    options.field =
        MTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::integer;
    if constexpr(std::is_same<void, ValueType>::value) throw ReaderException("You are reading the values of the matrix market file into a void array");
  } else if (field == "pattern") {
    options.field =
        MTXReader<IDType, NNZType, ValueType>::MTXFieldOptions::pattern;
  } else {
    throw ReaderException(
        "Illegal value for the 'field' option in matrix market header");
  }
  // parsing symmetry
  if (symmetry == "general") {
    options.symmetry =
        MTXReader<IDType, NNZType, ValueType>::MTXSymmetryOptions::general;
  } else if (symmetry == "symmetric") {
    options.symmetry =
        MTXReader<IDType, NNZType, ValueType>::MTXSymmetryOptions::symmetric;
  } else if (symmetry == "skew-symmetric") {
    options.symmetry = MTXReader<IDType, NNZType,
                                 ValueType>::MTXSymmetryOptions::skew_symmetric;
  } else if (symmetry == "hermitian") {
    options.symmetry =
        MTXReader<IDType, NNZType, ValueType>::MTXSymmetryOptions::hermitian;
    throw ReaderException("Matrix market reader does not currently support hermitian symmetry.");
  } else {
    throw ReaderException(
        "Illegal value for the 'symmetry' option in matrix market header");
  }
  return options;
}

template <typename IDType, typename NNZType, typename ValueType>
template <bool weighted>
format::COO<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadArrayIntoCOO() const {
  std::ifstream fin(filename_);
  // Ignore headers and comments:
  while (fin.peek() == '%')
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  // Declare variables: (check the types here)

  format::DimensionType M, N;

  fin >> M >> N;

  format::DimensionType total_values = M * N;
  IDType *long_rows = new IDType[total_values];
  IDType *long_cols = new IDType[total_values];
  ValueType *long_vals = nullptr;
  if constexpr (!weighted)
    long_vals= new ValueType[total_values];
  NNZType num_nnz = 0;
  for (format::DimensionType l = 0; l < total_values; l++) {
    ValueType w;
    fin >> w;

    if (w != 0){
      long_cols[num_nnz] = l/N;
      long_rows[num_nnz] = l%N;
      if constexpr (weighted)
        long_vals[num_nnz] = w;
      num_nnz++;

    }
    //nnz_counter += w != 0;
  }

  IDType *row = new IDType[num_nnz];
  IDType *col = new IDType[num_nnz];
  std::copy(long_rows, long_rows+num_nnz, row);
  std::copy(long_cols, long_cols+num_nnz, col);
  ValueType *vals = nullptr;
  if constexpr (weighted) {
    vals = new ValueType[num_nnz];
    std::copy(long_vals, long_vals+num_nnz, vals);
  }

  return new format::COO<IDType, NNZType, ValueType>(N, M, num_nnz, row, col, vals, format::kOwned);
}
template <typename IDType, typename NNZType, typename ValueType>
format::COO<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCOO() const {
  bool weighted = options_.field != MTXFieldOptions::pattern;
  if (options_.format == MTXFormatOptions::array){
    if (weighted)
      return this->ReadArrayIntoCOO<true>();
    else
      return this->ReadArrayIntoCOO<false>();
  } else if (options_.format == MTXFormatOptions::coordinate){
    if (weighted){
      if (options_.symmetry == MTXSymmetryOptions::general)
        if (this->convert_to_zero_index_)
          return this->ReadCoordinateIntoCOO<true, (int)MTXSymmetryOptions::general, true>();
        else
          return this->ReadCoordinateIntoCOO<true, (int)MTXSymmetryOptions::general, false>();
      else if (options_.symmetry == MTXSymmetryOptions::symmetric)
        if (this->convert_to_zero_index_)
        return this->ReadCoordinateIntoCOO<true, (int)MTXSymmetryOptions::symmetric, true>();
        else
        return this->ReadCoordinateIntoCOO<true, (int)MTXSymmetryOptions::symmetric, false>();
      else if (options_.symmetry == MTXSymmetryOptions::skew_symmetric)
        if (this->convert_to_zero_index_)
        return this->ReadCoordinateIntoCOO<true, (int)MTXSymmetryOptions::skew_symmetric, true>();
        else
        return this->ReadCoordinateIntoCOO<true, (int)MTXSymmetryOptions::skew_symmetric, false>();
      else
        throw ReaderException(
            "Can't read matrix market symmetry options besides general, symmetric, and skew_symmetric");
    } else {
      if (options_.symmetry == MTXSymmetryOptions::general)
        if (this->convert_to_zero_index_)
        return this->ReadCoordinateIntoCOO<false, (int)MTXSymmetryOptions::general, true>();
        else
        return this->ReadCoordinateIntoCOO<false, (int)MTXSymmetryOptions::general, false>();
      else if (options_.symmetry == MTXSymmetryOptions::symmetric)
        if (this->convert_to_zero_index_)
        return this->ReadCoordinateIntoCOO<false, (int)MTXSymmetryOptions::symmetric, true>();
        else
        return this->ReadCoordinateIntoCOO<false, (int)MTXSymmetryOptions::symmetric, false>();
      else if (options_.symmetry == MTXSymmetryOptions::skew_symmetric)
        if (this->convert_to_zero_index_)
        return this->ReadCoordinateIntoCOO<false, (int)MTXSymmetryOptions::skew_symmetric, true>();
        else
        return this->ReadCoordinateIntoCOO<false, (int)MTXSymmetryOptions::skew_symmetric, false>();
      else
        throw ReaderException(
            "Can't read matrix market symmetry options besides general, symmetric, and skew_symmetric");
    }
  } else {
    throw ReaderException("Can't read matrix market formats besides array and coordinate");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
format::Array<ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCoordinateIntoArray() const {
  std::ifstream fin(filename_);

  // Ignore headers and comments:
  while (fin.peek() == '%')
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  // Declare variables: (check the types here)

  // check that it has 1 row/column
  format::DimensionType M, N, L;

  fin >> M >> N >> L;
  if (M != 1 && N != 1) {
    throw ReaderException("Trying to read a 2D matrix with multiple rows "
                          "and multiple columns into dense array");
  }
  fin.close();
  auto coo = ReadCOO();
  auto n = coo->get_dimensions()[0];
  auto m = coo->get_dimensions()[1];
  auto coo_col = coo->get_col();
  auto coo_row = coo->get_row();
  auto coo_vals = coo->get_vals();
  auto num_nnz = coo->get_num_nnz();
  ValueType *vals = new ValueType[std::max<IDType>(n, m)]();
  IDType curr_row = 0;
  IDType curr_col = 0;
  for (IDType nnz = 0; nnz < num_nnz; nnz++) {
    vals[coo_col[nnz] + coo_row[nnz]] = coo_vals[nnz];
  }
  return new format::Array<ValueType>(std::max(n, m), vals,
                                      sparsebase::format::kOwned);
}

template <typename IDType, typename NNZType, typename ValueType>
template <bool weighted, int symm, bool conv_to_zero>
format::COO<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCoordinateIntoCOO() const {
  // Open the file:
  std::ifstream fin(filename_);

  if (fin.is_open()) {
    // Declare variables: (check the types here)
    format::DimensionType M, N, L;

    // Ignore headers and comments:
    while (fin.peek() == '%')
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    fin >> M >> N >> L;
    ValueType *vals = nullptr;
    if constexpr (symm == (int)MTXSymmetryOptions::general) {
      IDType *row = new IDType[L];
      IDType *col = new IDType[L];
      if constexpr (weighted) {
        if constexpr (!std::is_same_v<void, ValueType>) {
          vals = new ValueType[L];
          for (NNZType l = 0; l < L; l++) {
            IDType m, n;
            ValueType w;
            fin >> m >> n >> w;

            if constexpr (conv_to_zero) {
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

          if constexpr (conv_to_zero) {
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
    } else if constexpr (symm == (int)MTXSymmetryOptions::symmetric || symm == (int)MTXSymmetryOptions::skew_symmetric) {
      IDType *row = new IDType[L * 2];
      IDType *col = new IDType[L * 2];
      NNZType actual_nnzs = 0;
      IDType m, n;
      ValueType w;
      if constexpr (weighted) {
        if constexpr (!std::is_same_v<void, ValueType>) {
          vals = new ValueType[L * 2];
        }
      }
      for (NNZType l = 0; l < L; l++) {
        fin >> m >> n;
        if constexpr (weighted)
          fin >> w;

        if constexpr (conv_to_zero) {
          n--;
          m--;
        }
        row[actual_nnzs] = m;
        col[actual_nnzs] = n;
        if constexpr (weighted)
          vals[actual_nnzs] = w;
        actual_nnzs++;
        bool check_diagonal;
        if constexpr (symm == (int)MTXSymmetryOptions::skew_symmetric)
          check_diagonal = false;
        else
          check_diagonal = true;
        if (check_diagonal && m != n) {
          row[actual_nnzs] = n;
          col[actual_nnzs] = m;
          if constexpr (weighted)
            vals[actual_nnzs] = w;
          actual_nnzs++;
        }
      }
      IDType *actual_rows = row;
      IDType *actual_cols = col;
      ValueType *actual_vals = vals;
      if (symm == (int)MTXSymmetryOptions::symmetric && actual_nnzs != L * 2) {
        actual_rows = new IDType[actual_nnzs];
        actual_cols = new IDType[actual_nnzs];
        std::copy(row, row+actual_nnzs, actual_rows);
        std::copy(col, col+actual_nnzs, actual_cols);
        delete[] row;
        delete[] col;
        if constexpr (weighted) {
          actual_vals = new ValueType[actual_nnzs];
          std::copy(vals, vals+actual_nnzs, actual_vals);
          delete[] vals;
        }
      }
      auto coo = new format::COO<IDType, NNZType, ValueType>(
          M, N, actual_nnzs, actual_rows, actual_cols, actual_vals,
          format::kOwned);
      return coo;
    } else {
      throw ReaderException("Reader only supports general, symmetric, and skew-symmetric symmetry options");
    }
  } else {
    throw ReaderException("file does not exists!!");
  }
}
template <typename IDType, typename NNZType, typename ValueType>
format::Array<ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadArrayIntoArray() const {
  std::ifstream fin(filename_);
  // Ignore headers and comments:
  while (fin.peek() == '%')
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  // Declare variables: (check the types here)

  // check that it has 1 row/column
  format::DimensionType M, N;

  fin >> M >> N;

  if (M != 1 && N != 1) {
    throw ReaderException("Trying to read a 2D matrix with multiple rows "
                          "and multiple columns into dense array");
  }
  format::DimensionType total_values = M * N;
  // TODO
  // Currently num_nnz is defined all wrong. Once it's fixed add back nnz_counter
  //NNZType nnz_counter = 0;
  ValueType *vals = new ValueType[total_values];
  for (format::DimensionType l = 0; l < total_values; l++) {
    ValueType w;
    fin >> w;

    vals[l] = w;
    //nnz_counter += w != 0;
  }

  auto array = new format::Array<ValueType>(/*nnz_counter*/total_values, vals, format::kOwned);
  return array;
}

template <typename IDType, typename NNZType, typename ValueType>
format::Array<ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadArray() const {
  // check object
  if (options_.format == MTXFormatOptions::coordinate) {
      return ReadCoordinateIntoArray();
  } else if (options_.format == MTXFormatOptions::array) {
      return ReadArrayIntoArray();
  } else {
    throw ReaderException(
        "Wrong format value while reading matrix market file\n");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType> *
MTXReader<IDType, NNZType, ValueType>::ReadCSR() const {
  auto coo = ReadCOO();
  converter::ConverterOrderTwo<IDType, NNZType, ValueType> converterObj;
  context::CPUContext cpu_context;
  return converterObj.template Convert<format::CSR<IDType, NNZType, ValueType>>(
      coo, &cpu_context);
}

template <typename IDType, typename NNZType, typename ValueType>
MTXReader<IDType, NNZType, ValueType>::~MTXReader(){};

template <typename IDType, typename NNZType, typename ValueType>
PigoMTXReader<IDType, NNZType, ValueType>::PigoMTXReader(
    std::string filename, bool weighted, bool convert_to_zero_index)
    : filename_(filename), weighted_(weighted),
      convert_to_zero_index_(convert_to_zero_index) {}

//template <typename IDType, typename NNZType, typename ValueType>
//format::Array<ValueType> *
//PigoMTXReader<IDType, NNZType, ValueType>::ReadArray() const {
//  MTXReader<IDType, NNZType, ValueType> reader(filename_, weighted_);
//  return reader.ReadArray();
//}

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
        pigo_coo.nrows() - 1, pigo_coo.ncols() - 1, pigo_coo.m(), pigo_coo.x(),
        pigo_coo.y(), pigo_coo.w(), format::kOwned);
  } else {
    pigo::COO<IDType, IDType, IDType *, false, false, false, false, ValueType,
              ValueType *>
        pigo_coo(filename_, pigo::MATRIX_MARKET);
    coo = new format::COO<IDType, NNZType, ValueType>(
        pigo_coo.nrows() - 1, pigo_coo.ncols() - 1, pigo_coo.m(), pigo_coo.x(),
        pigo_coo.y(), nullptr, format::kOwned);
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
format::CSR<IDType, NNZType, ValueType> *
PigoMTXReader<IDType, NNZType, ValueType>::ReadCSR() const {
  format::COO<IDType, NNZType, ValueType> *coo = ReadCOO();
  utils::converter::ConverterOrderTwo<IDType, NNZType, ValueType> converter;
  std::cout << "nnz " << coo->get_num_nnz() << " dim " << coo->get_dimensions()[0] << " " << coo->get_dimensions()[1] << std::endl;
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
        coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), coo.w(),
        format::kOwned);
  } else {
    pigo::COO<IDType, IDType, IDType *, false, false, false, false, ValueType,
              ValueType *>
        coo(filename_, pigo::EDGE_LIST);
    return new format::COO<IDType, NNZType, ValueType>(
        coo.nrows(), coo.ncols(), coo.m(), coo.x(), coo.y(), nullptr,
        format::kOwned);
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
#endif

} // namespace io

} // namespace utils

} // namespace sparsebase
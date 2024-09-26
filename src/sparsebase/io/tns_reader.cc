#include "sparsebase/io/tns_reader.h"

#include <sstream>
#include <string>

#include "sparsebase/config.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
TNSReader<IDType, NNZType, ValueType>::TNSReader(std::string filename, bool store_values,
                                                 bool convert_to_zero_index)
    : filename_(filename), convert_to_zero_index_(convert_to_zero_index), store_values_(store_values) {
  std::ifstream fin(filename_);

  if (!fin.is_open())
    throw utils::ReaderException("Wrong tns file name\n");
}

template <typename IDType, typename NNZType, typename ValueType>
format::HigherOrderCOO<IDType, NNZType, ValueType>
    *TNSReader<IDType, NNZType, ValueType>::ReadHigherOrderCOO() const {
  std::ifstream fin(filename_);

  if (fin.is_open()) {

    NNZType L = 0;
    format::DimensionType N = 0;

    while (fin.peek() == '#')
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string line;
    std::getline(fin, line);

    std::stringstream ss(line);
    std::string z;
    while (ss >> z) N++;
    N--;

    while (fin.peek() != EOF) {
      if (fin.peek() != '#') {
        L++;
      }
      fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    L++;

    format::DimensionType *dimension = new format::DimensionType[N];

    IDType **indices = new IDType *[N];
    for (format::DimensionType i = 0; i < N; i++) indices[i] = new IDType[L];

    fin.clear();
    fin.seekg(0);

    ValueType *vals = nullptr;

    if (store_values_) {
      if constexpr (std::is_same_v<void, ValueType>) {
        throw ReaderException(
            "Value type can not be void if store_values option is true.");
      }

      vals = new ValueType[L];

      for (NNZType l = 0; l < L; l++) {
        while (fin.peek() == '#')
          fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        for (format::DimensionType j = 0; j < N; j++) {
          fin >> indices[j][l];

          if (dimension[j] < indices[j][l]) dimension[j] = indices[j][l];

          indices[j][l] -= convert_to_zero_index_;
        }

        fin >> vals[l];
      }
    } else {
      for (NNZType l = 0; l < L; l++) {
        while (fin.peek() == '#')
          fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        for (format::DimensionType j = 0; j < N; j++) {
          fin >> indices[j][l];

          if (dimension[j] < indices[j][l]) dimension[j] = indices[j][l];

          indices[j][l] -= convert_to_zero_index_;
        }

        fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      }
    }
    fin.close();

    auto higher_order_coo = new format::HigherOrderCOO<IDType, NNZType, ValueType>(N, dimension, L, indices, vals, format::kOwned);
    
    return higher_order_coo;
  }
}

template <typename IDType, typename NNZType, typename ValueType>
TNSReader<IDType, NNZType, ValueType>::~TNSReader(){};
#ifndef _HEADER_ONLY
#include "init/tns_reader.inc"
#endif
}  // namespace sparsebase::io
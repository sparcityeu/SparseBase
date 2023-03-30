#include <fstream>
#include <sstream>
#include "sparsebase/io/metis_graph_reader.h"
#include "sparsebase/config.h"
#include "sparsebase/io/reader.h"

namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
MetisGraphReader<IDType, NNZType, ValueType>::MetisGraphReader(
    std::string filename)
    : filename_(filename) {}

template <typename IDType, typename NNZType, typename ValueType>
sparsebase::object::Graph<IDType, NNZType, ValueType>
    *MetisGraphReader<IDType, NNZType, ValueType>::ReadGraph() const {
  IDType n, m;
  IDType *row;
  IDType *col;
  ValueType *val;
  int FMT = 0, NCON = 0;
  bool isEdgeWeighted = false, isVertexWeighted = false;
  std::ifstream infile(this->filename_);
  std::string line;
  if (infile.is_open()) {
    while (std::getline(infile, line)) {
      if (line[0] != '%') {
        std::istringstream iss(line);
        iss >> n >> m;
        m *= 2;
        if (iss >> FMT);
        if (iss >> NCON);
        if ((FMT == 1 || FMT == 11) && NCON == 0) NCON = 1;
        isEdgeWeighted = (FMT == 1 || FMT == 11);
        isVertexWeighted = (FMT >= 10 && NCON > 0);
        break;
      }
    }
    row = new IDType[m];
    col = new IDType[m];
    val = nullptr;
    IDType node = 0, neig, curr = 0;
    if constexpr (std::is_same_v<ValueType, void>) {
      //No vertex/edge weights
      while (std::getline(infile, line)) {
        if (line[0] != '%') {
          std::istringstream iss(line);
          ++node;
          while (iss >> neig) {
            row[curr] = node;
            col[curr] = neig;
            ++curr;
          }
        }
      }
      return new sparsebase::object::Graph<IDType, NNZType, ValueType>(
          new format::COO<IDType, NNZType, ValueType>(n, n, m, row, col,val)
              );
    } else {
      format::Array<ValueType>** vertexWeights = nullptr;
      if (isVertexWeighted) {
        vertexWeights = new format::Array<ValueType>*[n+1];

        //Nodes are 1-indexed, weights for node 0 set to 0's as default
        ValueType* tmp = new ValueType[NCON];
        for (int i = 0; i < NCON; ++i) tmp[i] = 0;
        vertexWeights[0] = new format::Array<ValueType>(NCON, tmp);
      }
      if (isEdgeWeighted) {
        val = new ValueType[m];
      }
      while (std::getline(infile, line)) {
        if (line[0] != '%') {
          std::istringstream iss(line);
          ++node;
          if (isVertexWeighted) {
            ValueType* tmp = new ValueType[NCON];
            for (int i = 0; i < NCON; ++i) iss >> tmp[i];
            vertexWeights[node] = new format::Array<ValueType>(NCON, tmp);
          }
          while (iss >> neig) {
            row[curr] = node;
            col[curr] = neig;
            if (isEdgeWeighted)
            {
              iss >> val[curr];
            }
            ++curr;
          }
        }
      }
      return new sparsebase::object::Graph<IDType, NNZType, ValueType>(
          new format::COO<IDType, NNZType, ValueType>(n, n, m, row, col,val),
              NCON, vertexWeights
          );
    }
  }else {
    throw utils::ReaderException("file does not exist!");
  }
}

template <typename IDType, typename NNZType, typename ValueType>
MetisGraphReader<IDType, NNZType, ValueType>::~MetisGraphReader(){};
#ifndef _HEADER_ONLY
#include "init/metis_graph_reader.inc"
#endif
}  // namespace sparsebase::io

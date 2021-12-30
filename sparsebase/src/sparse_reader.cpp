#include "sparsebase/sparse_reader.hpp"
#include "sparsebase/sparse_exception.hpp"
#include "sparsebase/sparse_format.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace sparsebase {
template <typename ID_t, typename NNZ_t, typename VAL_t>
SparseReader<ID_t, NNZ_t, VAL_t>::~SparseReader(){};

// Add weighted option with contexpr
//! Brief description
/*!
  Detailed description
  \param filename string
  \param _weighted bool
  \return vector of formats
*/
template <typename v_t, typename e_t, typename w_t>
UedgelistReader<v_t, e_t, w_t>::UedgelistReader(string filename, bool _weighted)
    : filename(filename), weighted(_weighted) {}
template <typename v_t, typename e_t, typename w_t>
SparseFormat<v_t, e_t, w_t> *
UedgelistReader<v_t, e_t, w_t>::read_sparseformat() const {
  return this->read_csr();
}
template <typename v_t, typename e_t, typename w_t>
CSR<v_t, e_t, w_t> *UedgelistReader<v_t, e_t, w_t>::read_csr() const {
  std::ifstream infile(this->filename);
  if (infile.is_open()) {
    v_t u, v;
    e_t edges_read = 0;
    v_t n = 0;

    std::vector<std::pair<v_t, v_t>> edges;
    // vertices are 0-based
    while (infile >> u >> v) {
      if (u != v) {
        edges.push_back(std::pair<v_t, v_t>(u, v));
        edges.push_back(std::pair<v_t, v_t>(v, u));

        n = max(n, u);
        n = max(n, v);

        edges_read++;
      }
    }
    n++;
    std::cout << "No vertices is " << n << endl;
    std::cout << "No read edges " << edges_read << endl;
    e_t m = edges.size();
    std::cout << "No edges is " << m << endl;

    sort(edges.begin(), edges.end(), sortedge);
    edges.erase(unique(edges.begin(), edges.end()), edges.end());

    // allocate the memory
    e_t *row_ptr = new e_t[n + 1];
    v_t *col = new v_t[m];
    v_t *tadj = new v_t[m];
    v_t *is = new v_t[m];

    // populate col and row_ptr
    memset(row_ptr, 0, sizeof(e_t) * (n + 1));
    int mt = 0;
    for (std::pair<v_t, v_t> &e : edges) {
      row_ptr[e.first + 1]++;
      is[mt] = e.first;
      col[mt++] = e.second;
    }

    for (e_t i = 1; i <= n; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    for (v_t i = 0; i < m; i++) {
      tadj[i] = row_ptr[col[i]]++;
    }
    for (e_t i = n; i > 0; i--) {
      row_ptr[i] = row_ptr[i - 1];
    }
    row_ptr[0] = 0;
    return new CSR<v_t, e_t, w_t>(n, n, row_ptr, col, nullptr);
  } else {
    throw invalid_argument("file does not exists!!");
  }
}
template <typename v_t, typename e_t, typename w_t>
bool UedgelistReader<v_t, e_t, w_t>::sortedge(const pair<v_t, v_t> &a,
                                              const pair<v_t, v_t> &b) {
  if (a.first == b.first) {
    return (a.second < b.second);
  } else {
    return (a.first < b.first);
  }
}
template <typename v_t, typename e_t, typename w_t>
UedgelistReader<v_t, e_t, w_t>::~UedgelistReader(){};
template <typename v_t, typename e_t, typename w_t>
SparseFormat<v_t, e_t, w_t> *
MTXReader<v_t, e_t, w_t>::read_sparseformat() const {
  return this->read_coo();
}

template <typename v_t, typename e_t, typename w_t>
MTXReader<v_t, e_t, w_t>::MTXReader(string filename, bool _weighted)
    : filename(filename), weighted(_weighted) {}

template <typename v_t, typename e_t, typename w_t>
COO<v_t, e_t, w_t> *MTXReader<v_t, e_t, w_t>::read_coo() const {
  // Open the file:
  std::ifstream fin(filename);

  // Declare variables: (check the types here)
  v_t M, N, L;

  // Ignore headers and comments:
  while (fin.peek() == '%')
    fin.ignore(std::numeric_limits<streamsize>::max(), '\n');

  fin >> M >> N >> L;

  v_t *row = new v_t[L];
  v_t *col = new v_t[L];
  if constexpr (!std::is_same_v<void, w_t>) {
    if (weighted) {
      w_t *vals = new w_t[L];
      for (e_t l = 0; l < L; l++) {
        v_t m, n;
        w_t w;
        fin >> m >> n >> w;
        row[l] = n - 1;
        col[l] = m - 1;
        vals[l] = w;
      }

      auto coo = new COO<v_t, e_t, w_t>(M, N, L, row, col, vals);
      return coo;
    } else {
      // TODO: Add an exception class for this
      throw SparseReaderException(
          "Weight type for weighted graphs can not be void");
    }
  } else {
    for (e_t l = 0; l < L; l++) {
      v_t m, n;
      fin >> m >> n;
      row[l] = m - 1;
      col[l] = n - 1;
    }

    auto coo = new COO<v_t, e_t, w_t>(M, N, L, row, col, nullptr);
    return coo;
  }
}

template <typename v_t, typename e_t, typename w_t>
MTXReader<v_t, e_t, w_t>::~MTXReader(){};

template class MTXReader<unsigned int, unsigned int, unsigned int>;
template class UedgelistReader<unsigned int, unsigned int, unsigned int>;

template class MTXReader<unsigned int, unsigned int, void>;
template class UedgelistReader<unsigned int, unsigned int, void>;

} // namespace sparsebase

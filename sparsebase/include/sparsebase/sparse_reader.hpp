#ifndef _SPARSEREADER_HPP
#define _SPARSEREADER_HPP

#include "sparse_format.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

using namespace std;

namespace sparsebase {

template <typename ID_t, typename NNZ_t, typename VAL_t> class SparseReader {
public:
  virtual ~SparseReader();
};

template <class v_t, typename e_t, typename w_t> class ReadsSparseFormat {
public:
  virtual SparseFormat<v_t, e_t, w_t> *read_sparseformat() const = 0;
};

template <class v_t, typename e_t, typename w_t> class ReadsCSR {
public:
  virtual CSR<v_t, e_t, w_t> *read_csr() const = 0;
};

template <class v_t, typename e_t, typename w_t> class ReadsCOO {
public:
  virtual COO<v_t, e_t, w_t> *read_coo() const = 0;
};
// Add weighted option with contexpr
template <typename v_t, typename e_t, typename w_t>
class UedgelistReader : public SparseReader<v_t, e_t, w_t>,
                        public ReadsCSR<v_t, e_t, w_t>,
                        public ReadsSparseFormat<v_t, e_t, w_t> {
public:
  UedgelistReader(string filename, bool _weighted = false);
  CSR<v_t, e_t, w_t> *read_csr() const;
  SparseFormat<v_t, e_t, w_t> *read_sparseformat() const;
  virtual ~UedgelistReader();

private:
  static bool sortedge(const pair<v_t, v_t> &a, const pair<v_t, v_t> &b);
  string filename;
  bool weighted;
};

template <typename v_t, typename e_t, typename w_t>
class MTXReader : public SparseReader<v_t, e_t, w_t>,
                  public ReadsCOO<v_t, e_t, w_t>,
                  public ReadsSparseFormat<v_t, e_t, w_t> {
public:
  MTXReader(string filename, bool _weighted = false);
  COO<v_t, e_t, w_t> *read_coo() const;
  SparseFormat<v_t, e_t, w_t> *read_sparseformat() const;
  virtual ~MTXReader();

private:
  string filename;
  bool weighted;
};

} // namespace sparsebase

#endif

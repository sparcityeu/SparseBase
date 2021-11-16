#ifndef _SPARSEREADER_HPP
#define _SPARSEREADER_HPP

#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include "SparseFormat.hpp"

using namespace std;

namespace sparsebase{

  template<typename ID_t, typename NNZ_t>
  class SparseReader{
    public:
      virtual ~SparseReader();
      virtual vector<SparseFormat<ID_t, NNZ_t> *> read() = 0;
  };

// Add weighted option with contexpr
  template<typename v_t, typename e_t, typename w_t>
    class UedgelistReader : public SparseReader<v_t, e_t>{
      public:
        UedgelistReader(string filename, bool _weighted=false);
        vector<SparseFormat<v_t, e_t> *> read();
      private:
        static bool sortedge(const pair<v_t,v_t> &a,
            const pair<v_t,v_t> &b);
        string filename;
        bool weighted;
        virtual ~UedgelistReader();
    };

}

#endif

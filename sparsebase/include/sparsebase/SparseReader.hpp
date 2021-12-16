#ifndef _SPARSEREADER_HPP
#define _SPARSEREADER_HPP

#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include "SparseFormat.hpp"

using namespace std;

namespace sparsebase{

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class SparseReader{
    public:
      virtual ~SparseReader();
      virtual vector<SparseFormat<ID_t, NNZ_t, VAL_t> *> read() = 0;
  };

// Add weighted option with contexpr
  template<typename v_t, typename e_t, typename w_t>
    class UedgelistReader : public SparseReader<v_t, e_t, w_t>{
      public:
        UedgelistReader(string filename, bool _weighted=false);
        vector<SparseFormat<v_t, e_t, w_t> *> read();
      private:
        static bool sortedge(const pair<v_t,v_t> &a,
            const pair<v_t,v_t> &b);
        string filename;
        bool weighted;
        virtual ~UedgelistReader();
    };

    template<typename v_t, typename e_t, typename w_t>
    class MTXReader : public SparseReader<v_t, e_t, w_t>{
    public:
        MTXReader(string filename, bool _weighted=false);
        vector<SparseFormat<v_t, e_t, w_t> *> read();
    private:
        string filename;
        bool weighted;
        virtual ~MTXReader();
    };



}

#endif
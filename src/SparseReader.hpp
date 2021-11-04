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
      virtual ~SparseReader(){};
      virtual vector<SparseFormat<ID_t, NNZ_t> *> read() = 0;
  };

// Add weighted option with contexpr
  template<typename v_t, typename e_t, typename w_t>
    class UedgelistReader : public SparseReader<v_t, e_t>{
      public:
        UedgelistReader(string filename, bool _weighted=false): filename(filename), weighted(_weighted) {}
        vector<SparseFormat<v_t, e_t> *> read(){
          ifstream infile(this->filename);
          if(infile.is_open()) {
            v_t u, v;
            e_t edges_read = 0;
            v_t n = 0;

            vector< std::pair<v_t, v_t> > edges;
            //vertices are 0-based 
            while (infile >> u >> v) {
              if(u != v) {
                edges.push_back(std::pair<v_t, v_t>(u, v));
                edges.push_back(std::pair<v_t, v_t>(v, u));

                n = max(n, u);
                n = max(n, v);

                edges_read++;
              }
            }
            n++;
            cout << "No vertices is " << n << endl;
            cout << "No read edges " << edges_read << endl;
            e_t m = edges.size();
            cout << "No edges is " << m << endl;

            sort(edges.begin(), edges.end(), sortedge);
            edges.erase( unique( edges.begin(), edges.end() ), edges.end() );

            //allocate the memory
            e_t * xadj = new e_t[n + 1];
            v_t * adj = new v_t[m];
            v_t * tadj = new v_t[m];
            v_t * is = new v_t[m];

            //populate adj and xadj
            memset(xadj, 0, sizeof(e_t) * (n + 1));
            int mt = 0;
            for(std::pair<v_t, v_t>& e : edges) {
              xadj[e.first + 1]++;
              is[mt] = e.first;
              adj[mt++] = e.second;
            }

            for(e_t i = 1; i <= n; i++) {
              xadj[i] += xadj[i-1];
            }

            for(v_t i = 0; i < m; i++) {
              tadj[i] = xadj[adj[i]]++;
            }
            for(e_t i = n; i > 0; i--) {
              xadj[i] = xadj[i-1];
            }
            xadj[0] = 0;
            return {new CSR<v_t, e_t, void>(n, n, xadj, adj, nullptr)}; 
          } else {
            throw invalid_argument("file does not exists!");
          }
        }
      private:
        static bool sortedge(const pair<v_t,v_t> &a,
            const pair<v_t,v_t> &b) {
          if(a.first == b.first) {
            return (a.second < b.second);
          } else {
            return (a.first < b.first);
          }
        }
        string filename;
        bool weighted;
        virtual ~UedgelistReader(){};
    };

}

#endif

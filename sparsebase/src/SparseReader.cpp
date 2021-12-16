#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseReader.hpp"
#include "sparsebase/SparseException.hpp"

namespace sparsebase
{
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
  UedgelistReader<v_t, e_t, w_t>::UedgelistReader(string filename, bool _weighted) : filename(filename), weighted(_weighted) {}
  template <typename v_t, typename e_t, typename w_t>
  std::vector<SparseFormat<v_t, e_t, w_t> *> UedgelistReader<v_t, e_t, w_t>::read()
  {
    std::ifstream infile(this->filename);
    if (infile.is_open())
    {
      v_t u, v;
      e_t edges_read = 0;
      v_t n = 0;

      std::vector<std::pair<v_t, v_t> > edges;
      //vertices are 0-based
      while (infile >> u >> v)
      {
        if (u != v)
        {
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

      //allocate the memory
      e_t *xadj = new e_t[n + 1];
      v_t *adj = new v_t[m];
      v_t *tadj = new v_t[m];
      v_t *is = new v_t[m];

      //populate adj and xadj
      memset(xadj, 0, sizeof(e_t) * (n + 1));
      int mt = 0;
      for (std::pair<v_t, v_t> &e : edges)
      {
        xadj[e.first + 1]++;
        is[mt] = e.first;
        adj[mt++] = e.second;
      }

      for (e_t i = 1; i <= n; i++)
      {
        xadj[i] += xadj[i - 1];
      }

      for (v_t i = 0; i < m; i++)
      {
        tadj[i] = xadj[adj[i]]++;
      }
      for (e_t i = n; i > 0; i--)
      {
        xadj[i] = xadj[i - 1];
      }
      xadj[0] = 0;
      return {new CSR<v_t, e_t, void>(n, n, xadj, adj, nullptr)};
    }
    else
    {
      throw invalid_argument("file does not exists!!");
    }
  }
  template <typename v_t, typename e_t, typename w_t>
  bool UedgelistReader<v_t, e_t, w_t>::sortedge(const pair<v_t, v_t> &a,
                                                       const pair<v_t, v_t> &b)
  {
    if (a.first == b.first)
    {
      return (a.second < b.second);
    }
    else
    {
      return (a.first < b.first);
    }
  }
  template <typename v_t, typename e_t, typename w_t>
  UedgelistReader<v_t, e_t, w_t>::~UedgelistReader(){};


    template <typename v_t, typename e_t, typename w_t>
    MTXReader<v_t, e_t, w_t>::MTXReader(string filename, bool _weighted) : filename(filename), weighted(_weighted) {}

    template <typename v_t, typename e_t, typename w_t>
    std::vector<SparseFormat<v_t, e_t, w_t> *> MTXReader<v_t, e_t, w_t>::read() {
        // Open the file:
        std::ifstream fin(filename);

        // Declare variables: (check the types here)
        v_t M, N, L;

        // Ignore headers and comments:
        while (fin.peek() == '%') fin.ignore(2048, '\n');

        fin >> M >> N >> L;

        v_t* adj = new v_t[L];
        v_t* is = new v_t[L];
        if constexpr(!std::is_same_v<void,w_t>) {
            if(weighted){
                w_t* vals = new w_t[L];
                for (int l = 0; l < L; l++) {
                    int m, n;
                    w_t w;
                    fin >> m >> n >> w;
                    adj[l] = m-1;
                    is[l] = n-1;
                    vals[l] = w;
                }

                auto coo = new COO<v_t,e_t,w_t>(M,N,L,adj,is,vals);
                return vector<SparseFormat<v_t, e_t, w_t> *>(1,coo);
            } else {
                // TODO: Add an exception class for this
                throw SparseReaderException("Weight type for weighted graphs can not be void");
            }

        } else {
            for (int l = 0; l < L; l++) {
                int m, n;
                fin >> m >> n;
                adj[l] = m-1;
                is[l] = n-1;
            }

            auto coo = new COO<v_t, e_t, w_t>(M,N,L,adj,is,nullptr);
            return vector<SparseFormat<v_t, e_t, w_t> *>(1,coo);
        }
    }

    template <typename v_t, typename e_t, typename w_t>
    MTXReader<v_t, e_t, w_t>::~MTXReader(){};



        template class MTXReader<unsigned int, unsigned int, void>;
        template class UedgelistReader<unsigned int, unsigned int, void>;

}

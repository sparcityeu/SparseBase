#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include "sparsebase/SparseFormat.hpp"
#include "sparsebase/SparseReader.hpp"

namespace sparsebase
{
  template <typename ID_t, typename NNZ_t>
  SparseReader<ID_t, NNZ_t>::~SparseReader(){};

  // Add weighted option with contexpr
  template <typename v_t, typename e_t, typename w_t>
  UedgelistReader<v_t, e_t, w_t>::UedgelistReader(string filename, bool _weighted) : filename(filename), weighted(_weighted) {}
  template <typename v_t, typename e_t, typename w_t>
  std::vector<SparseFormat<v_t, e_t> *> UedgelistReader<v_t, e_t, w_t>::read()
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

  template class UedgelistReader<unsigned int, unsigned int, void>;
}

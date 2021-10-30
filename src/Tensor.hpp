#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

namespace sparsebase{

  // TENSORS

  class Tensor{
    public:
      virtual ~Tensor(){};
      virtual int get_rank() = 0;
  };

  //abstract class
  class AbstractTensor : public Tensor{
    public:
      unsigned int order;
      //initialize order in the constructor
      AbstractTensor(unsigned int order): order(order) {}
      virtual ~AbstractTensor(){};
      virtual int get_rank() = 0;
  };

  class CSF : public AbstractTensor{
    public:
      CSF(unsigned int order): AbstractTensor(order) {cout<<"CSF created."<< endl;}
    private:
      virtual ~CSF(){}; 
      virtual int get_rank(){return 0;}
  };

  template<typename i_t, typename nnz_t> 
  class CSR : public AbstractTensor{
    public:
      CSR(): AbstractTensor(2) {cout<<"CSR created."<< endl;}
      CSR(nnz_t * xadj, i_t * adj, i_t n, nnz_t m): AbstractTensor(2), xadj(xadj), adj(adj), n(n), m(m) {cout<<"CSR created."<< endl;}
    private:
      nnz_t * xadj;
      i_t * adj;
      i_t n;
      nnz_t m;
      virtual ~CSR(){}; 
      virtual int get_rank(){return 0;}
  };

  //template<typename i_t, typename c_t>
  class COO : public AbstractTensor{
    public:
      COO(unsigned int order): AbstractTensor(order) {cout<<"COO created."<< endl;}
    private:
      virtual ~COO(){}; 
      virtual int get_rank(){return 0;}
  };

  // TENSOR READERS

  class TensorReader{
    public:
      virtual ~TensorReader(){};
      virtual Tensor * read() = 0;
  };

  template<typename i_t, typename nnz_t>
    class MtxReader : public TensorReader{
      public:
        MtxReader(string filename): TensorReader(), filename(filename) {}
      private:
        static bool sortedge(const pair<i_t,i_t> &a,
            const pair<i_t,i_t> &b) {
          if(a.first == b.first) {
            return (a.second < b.second);
          } else {
            return (a.first < b.first);
          }
        }
        Tensor * read(){
          ifstream infile(this->filename);
          if(infile.is_open()) {
            i_t u, v;
            nnz_t edges_read = 0;
            i_t n = 0;

            vector< std::pair<i_t, i_t> > edges;
            //vertices are 0-based 
            while (infile >> u >> v) {
              if(u != v) {
                edges.push_back(std::pair<i_t, i_t>(u, v));
                edges.push_back(std::pair<i_t, i_t>(v, u));

                n = max(n, u);
                n = max(n, v);

                edges_read++;
              }
            }
            n++;
            cout << "No vertices is " << n << endl;
            cout << "No read edges " << edges_read << endl;
            nnz_t m = edges.size();
            cout << "No edges is " << m << endl;

            sort(edges.begin(), edges.end(), sortedge);
            edges.erase( unique( edges.begin(), edges.end() ), edges.end() );

            //allocate the memory
            nnz_t * xadj = new nnz_t[n + 1];
            i_t * adj = new i_t[m];
            i_t * tadj = new i_t[m];
            i_t * is = new i_t[m];

            //populate adj and xadj
            memset(xadj, 0, sizeof(nnz_t) * (n + 1));
            int mt = 0;
            for(std::pair<i_t, i_t>& e : edges) {
              xadj[e.first + 1]++;
              is[mt] = e.first;
              adj[mt++] = e.second;
            }

            for(nnz_t i = 1; i <= n; i++) {
              xadj[i] += xadj[i-1];
            }

            for(i_t i = 0; i < m; i++) {
              tadj[i] = xadj[adj[i]]++;
            }
            for(nnz_t i = n; i > 0; i--) {
              xadj[i] = xadj[i-1];
            }
            xadj[0] = 0;
            return new CSR<i_t, nnz_t>(xadj, adj, n, m); 
          } else {
            throw invalid_argument("file does not exists!");
          }
        }
        string filename;
        virtual ~MtxReader(){};
    };

  // TENSOR CREATORS

  class TensorCreator{ //could be improved
    public:
      virtual ~TensorCreator(){};
      virtual Tensor * create(unsigned int order = 2) = 0;
  };

  class CSFCreator : public TensorCreator{
    public:
      CSFCreator(){}
      ~CSFCreator(){}
      Tensor * create(unsigned int order){
        return new CSF(order);
      }
  };

  template<typename i_t, typename nnz_t>
    class CSRCreator : public TensorCreator{
      public:
        CSRCreator(){}
        ~CSRCreator(){}
        Tensor * create(unsigned int order = 2){
          return new CSR<i_t, nnz_t>();
        }
        Tensor * create(string filename, unsigned int order = 2){
          TensorReader * reader = new MtxReader<i_t, nnz_t>(filename);
          return reader->read();
        }
    };
}

#endif

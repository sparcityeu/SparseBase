#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

namespace sparsebase{
  enum Format{
    CSR_f,
    COO_f
  };
  // TENSORS

  template<typename ID_t, typename NNZ_t>
  class SparseFormat{
    public:
      virtual ~SparseFormat(){};
      virtual unsigned int get_order() = 0;
      virtual Format get_format() = 0;
      virtual std::vector<ID_t> get_dimensions() = 0;
      virtual NNZ_t get_num_nnz() = 0;
  };

  //abstract class
  template<typename ID_t, typename NNZ_t>
  class AbstractSparseFormat : public SparseFormat<ID_t, NNZ_t>{
    public:
      //initialize order in the constructor
      AbstractSparseFormat() {}
      virtual ~AbstractSparseFormat(){};
      unsigned int get_order(){
        return order;
      }
      Format get_format(){
        return format;
      }
      std::vector<ID_t>get_dimensions(){
        return dimension;
      }
      NNZ_t get_num_nnz(){
        return nnz;
      }
      unsigned int order;
      std::vector<ID_t> dimension;
      Format format;
      NNZ_t nnz;
  };

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class COO : public AbstractSparseFormat<ID_t, NNZ_t>{
    public:
      COO() {
        this->order = 2;
        this->format = Format::COO_f;
        this->dimension = std::vector<ID_t>(2,0);
        this->nnz = 0;
        adj = nullptr;
        vals = nullptr;
        }
        COO(ID_t _n, ID_t _m, NNZ_t _nnz, ID_t * _adj, ID_t* _is, VAL_t* _vals){
          adj = _adj;
          is = _is;
          vals = _vals;
          this->nnz = _nnz;
          this->format = Format::CSR_f;
          this->order = 2;
          this->dimension = {_n, _m};
        }
      virtual ~COO(){}; 
    private:
      ID_t * adj;
      ID_t * is;
      VAL_t * vals;
  };
  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class CSR : public AbstractSparseFormat<ID_t, NNZ_t>{
    public:
      CSR() {
        this->order = 2;
        this->format = Format::CSR_f;
        this->dimension = std::vector<ID_t>(2,0);
        this->nnz = 0;
        adj = nullptr;
        xadj = nullptr;
        vals = nullptr;
        }
        CSR(ID_t _n, ID_t _m, NNZ_t * _xadj, ID_t * _adj, VAL_t* _vals){
          xadj = _xadj;
          adj = _adj;
          vals = _vals;
          this->format = Format::CSR_f;
          this->order = 2;
          this->dimension = {_n, _m};
          this->nnz = xadj[this->dimension[0]];
        }
      virtual ~CSR(){}; 
      ID_t * adj;
      NNZ_t * xadj;
      VAL_t * vals;
  };

  template<typename ID_t, typename NNZ_t, typename VAL_t>
  class CSF : public AbstractSparseFormat<ID_t, NNZ_t>{
    public:
      CSF(unsigned int order) {
        //init CSF
      }
      virtual ~CSF(){}; 
      NNZ_t ** ind;
      VAL_t * vals;
  };

/*
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
*/
}
#endif

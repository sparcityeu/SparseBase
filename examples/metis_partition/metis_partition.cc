#include <sparsebase/sparsebase.h>
#include <iostream>

using namespace sparsebase;
using namespace std;

#define TYPES int, int, float

int main(int argc, char** argv){

  if (argc < 2) {
    cout << "Usage: ./degree_order <uedgelist_file>\n";
    cout << "Hint: You can use the edgelist: examples/data/com-dblp.uedgelist\n";
    return 1;
  }

  cout << "F t re  s sp r e!" << endl;
  string file_name = argv[1];
  context::CPUContext cpu_context;

  cout << "Reading inputs..." << endl;
  format::COO<TYPES>* coo = utils::io::EdgeListReader<TYPES>(file_name).ReadCOO();
  format::CSR<TYPES>* csr = utils::io::EdgeListReader<TYPES>(file_name).ReadCSR();

  cout << "Dimensions" << endl;
  cout << csr->get_dimensions()[0] << endl;
  cout << csr->get_dimensions()[1] << endl;

  cout << "Setting partition params..." << endl;
  preprocess::MetisGraphPartition<TYPES> metis;
  preprocess::MetisGraphPartition<TYPES>::MetisParams params;
  params.seed = 12;
  params.ufactor = 50;
  params.rtype = preprocess::MetisGraphPartition<TYPES>::METIS_RTYPE_GREEDY;

  cout << "Partitioning CSR..." << endl;
  auto* res2 = metis.Partition(csr, &params, {&cpu_context}, false);

  cout << "Partitioning COO..." << endl;
  auto* res1 = metis.Partition(coo, &params, {&cpu_context}, true);


  cout << "Comparing results..." << endl;
  for(int i=0; i<10; i++){
    cout << "Vertex: " << i
         << " Partition COO: " << res1[i]
         << " Partition CSR: " << res2[i] << endl;

    if(res1[i] != res2[i]){
      cerr << "Partition is inconsistent!" << endl;
      return 1;
    }
  }

  cout << "Applying partition..." << endl;
  preprocess::ApplyGraphPartition<TYPES> applyGraphPartition(res1, 2);
  auto formats = applyGraphPartition.GetTransformation(csr, {&cpu_context}, true);

  int index = 0;
  for(auto& format : formats){
    cout << "Dimensions of " << index++ << endl;
    cout << format->get_dimensions()[0] << endl;
    cout << format->get_dimensions()[1] << endl;
  }

  return 0;
}
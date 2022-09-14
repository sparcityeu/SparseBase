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

  format::CSR<TYPES>* csr = utils::io::EdgeListReader<TYPES>(file_name).ReadCSR();

  preprocess::MetisPartition<TYPES> metis;
  auto* res = metis.Partition(csr, {&cpu_context});

  for(int i=0; i<10; i++){
    cout << "Vertex: " << i << " Partition: " << res[i] << endl;
  }

  return 0;
}
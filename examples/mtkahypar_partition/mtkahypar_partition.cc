/*
Mt-KaHyPar (Multi-Threaded Karlsruhe Hypergraph Partitioner) is a shared-memory
multilevel graph and hypergraph partitioner equipped with parallel
implementations of techniques used in the best sequential partitioning
algorithms. Mt-KaHyPar can partition extremely large hypergraphs very fast and
with high quality.

The highest-quality configuration of Mt-KaHyPar computes partitions that are on
par with those produced by the best sequential partitioning algorithms, while
being almost an order of magnitude faster with only ten threads (e.g., when
compared to KaFFPa or KaHyPar)
*/

#include "sparsebase/partition/mtkahypar_partition.h"

#include <sparsebase/sparsebase.h>

#include <iostream>
#include <string>
#include <vector>

#include "sparsebase/io/edge_list_writer.h"
using namespace sparsebase;
using namespace std;

bool HasEnding(std::string const& fullString, std::string const& ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
}

#define TYPES int, int, float

int main(int argc, char** argv) {
  if (argc < 1) {
    cout << "Usage: ./mtkahypar_partition <uedgelist_file>\n";
    cout << "Hint: You can use the edgelist: "
            "../../examples/data/com-dblp.uedgelist\n";
    return 1;
  }

  cout << "Welcome To MT-KaHyPar!" << endl;
  string file_name = argv[1];
  context::CPUContext cpu_context;

  cout << "Choose reader..." << endl;
  bool is_mtx = HasEnding(file_name, ".mtx");

  format::COO<TYPES>* coo;
  format::CSR<TYPES>* csr;

  if (is_mtx) {
    cout << "mtx file detected, mtx reader chosen" << endl;
    auto reader = std::make_shared<io::MTXReader<TYPES>>(file_name);

    cout << "Reading inputs..." << endl;
    coo = reader->ReadCOO();
    csr = reader->ReadCSR();
  } else {
    cout << "uedgelist file detected, edge list reader chosen" << endl;
    auto reader = std::make_shared<io::EdgeListReader<TYPES>>(file_name);

    cout << "Reading inputs..." << endl;
    coo = reader->ReadCOO();
    csr = reader->ReadCSR();
  }

  cout << "Setting partition params..." << endl;
  partition::MtkahyparPartition<TYPES> mtkahypar;
  partition::MtkahyparPartitionParams params;

  cout << "Partitioning CSR..." << endl;
  auto* res2 = mtkahypar.Partition(csr, &params, {&cpu_context}, false);

  cout << "Partitioning COO..." << endl;
  auto* res1 = mtkahypar.Partition(coo, &params, {&cpu_context}, true);

  cout << "Comparing results..." << endl;
  for (int i = 0; i < 100; i++) {
    cout << "Vertex: " << i << " Partition COO: " << res1[i]
         << " Partition CSR: " << res2[i] << endl;

    if (res1[i] != res2[i]) {
      cerr << "Partition is inconsistent!" << endl;
      return 1;
    }
  }

  cout << "Partition is consistent!" << endl;

  return 0;
}

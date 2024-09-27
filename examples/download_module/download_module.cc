#include <sparsebase/io/download_extract.h>

#include <iostream>
#include <string>

#include "sparsebase/format/format.h"

using namespace std;

int main(int argc, char *argv[]) {
  unordered_map<string, sparsebase::format::Format *> alternative_res;

  sparsebase::io::SuiteSparseDownloader ssd(
      "https://suitesparse-collection-website.herokuapp.com/MM/Meszaros/"
      "p010.tar.gz");
  alternative_res = ssd.download();

  for (auto &res : alternative_res) {
    auto format = res.second->get_dimensions()[0];
    cout << res.first << "  --  " << format << "\n";
  }

  sparsebase::io::SnapDownloader sd(
      "http://snap.stanford.edu/data/p2p-Gnutella05.txt.gz");

  alternative_res = sd.download();

  for (auto &res : alternative_res) {
    auto format = res.second->get_dimensions()[0];
    cout << res.first << " -- " << format << "\n";
  }

  sparsebase::io::NetworkRepositoryDownloader nrd(
      "https://nrvis.com/download/data/asn/"
      "aves-barn-swallow-contact-network.zip");
  alternative_res = nrd.download();

  for (auto &res : alternative_res) {
    auto format = res.second->get_dimensions()[0];
    cout << res.first << "  --  " << format << "\n";
  }

  sparsebase::io::NetworkRepositoryDownloader nrd2(
      "https://nrvis.com/download/data/misc/1138_bus.zip");
  alternative_res = nrd2.download();

  for (auto &res : alternative_res) {
    auto format = res.second->get_dimensions()[0];
    cout << res.first << "  --  " << format << "\n";
  }
}
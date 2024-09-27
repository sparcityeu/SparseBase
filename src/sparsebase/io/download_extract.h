#ifndef SPARSEBASE_SPARSEBASE_UTILS_IO_DOWNLOADEXTRACTMODULE_H_
#define SPARSEBASE_SPARSEBASE_UTILS_IO_DOWNLOADEXTRACTMODULE_H_

#include <string>
#include <unordered_map>

#include "sparsebase/format/format.h"
namespace sparsebase {

namespace io {

class DownloadExtractModule {
 public:
  std::string downloadUrl, filePath;
  DownloadExtractModule(std::string fileUrl);
  std::string returnFileName();
  static int returnFileType(std::string filePath);
};

class Downloader : public DownloadExtractModule {
 public:
  std::string deneme;
  Downloader();
  static bool download(const std::string &url, const std::string &filePath);
};

class Extractor : public DownloadExtractModule {
 public:
  static int extract(const std::string &filename);
};

class SuiteSparseDownloader {
 public:
  std::string matrixName;
  std::string matrixLocation;
  SuiteSparseDownloader(std::string matrixName,
                        std::string location = "DEFAULT");

  std::unordered_map<std::string, format::Format *> download();
};

class NetworkRepositoryDownloader {
 public:
  std::string matrixName;
  std::string matrixLocation;
  NetworkRepositoryDownloader(std::string matrixName,
                              std::string location = "DEFAULT");
  std::unordered_map<std::string, format::Format *> download();
  static void changePermission(std::string &path);
};

class SnapDownloader {
 public:
  std::string matrixName;
  std::string matrixLocation;
  SnapDownloader(std::string matrixName, std::string location = "DEFAULT");

  std::unordered_map<std::string, format::Format *> download();
  void cleanFile(std::string path);
};

class FrosttDownloader {
 public:
  std::string matrixName;
  std::string matrixLocation;
  FrosttDownloader(std::string matrixName, std::string location = "DEFAULT");

  std::unordered_map<std::string, format::Format *> download();
};

}  // namespace io

}  // namespace sparsebase
#ifdef _HEADER_ONLY
#include "sparsebase/utils/io/download_extract.cc"
#endif
#endif  // SPARSEBASE_SPARSEBASE_UTILS_IO_DOWNLOADEXTRACTMODULE_H_
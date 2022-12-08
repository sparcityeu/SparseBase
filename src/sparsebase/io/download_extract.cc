#include "sparsebase/io/download_extract.h"

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "sparsebase/format/format.h"
#include "sparsebase/io/edge_list_reader.h"
#include "sparsebase/io/mtx_reader.h"
#include "sparsebase/io/reader.h"
#include "sparsebase/utils/utils.h"

#define WIN 0
#define MACOSX 0
#define LINUX 0
#define UNIX 0

#ifdef _WIN64
#define WIN 1
#elif _WIN32
#define WIN 1
#elif __APPLE__
#define MACOSX 1
#elif __linux
#define LINUX 1
#elif __unix
#define UNIX 1
#endif

typedef unsigned int ui;

namespace sparsebase {

namespace io {
///
DownloadExtractModule::DownloadExtractModule(std::string fileUrl)
    : downloadUrl(fileUrl) {}

std::string DownloadExtractModule::returnFileName() {
  std::string url = this->downloadUrl;

  int i = url.size() - 1;

  for (; i >= 0; i--) {
    if (url[i] == '/') {
      break;
    }
  }
  this->filePath = url.substr(i + 1, url.size() - 1);
  return this->filePath;
}

int DownloadExtractModule::returnFileType(std::string filePath) {
  int i = 0;
  std::string fileExtension;
  for (; i < filePath.size(); i++) {
    if (filePath[i] == '.') break;
  }
  fileExtension = filePath.substr(i + 1, filePath.size() - 1);

  if (fileExtension == "tar.gz") {
    return 1;
  } else if (fileExtension == "gz" || fileExtension == "txt.gz") {
    return 2;
  } else if (fileExtension == "zip") {
    return 3;
  } else if (fileExtension == "7z") {
    return 4;
  } else if (fileExtension == "tgz") {
    return 5;
  } else {
    throw std::invalid_argument("Unsupported archive type!");
  }

  // if it is not a supported file type
  return -1;
}

bool Downloader::download(const std::string &url, const std::string &filePath) {
  std::system(("curl -L -o " + filePath + " " + url).c_str());

  return true;
}

int Extractor::extract(const std::string &filename) {
  std::string folderName;

  for (int i = 0; i < filename.size(); i++) {
    if (filename[i] == '.') {
      folderName = filename.substr(0, i);
      break;
    }
  }

  int res = returnFileType(filename);
  if (res == 1 || res == 5) {
    std::system(("tar xzvf " + filename).c_str());
  } else if (res == 2) {
    std::system(("mkdir "));
    std::system(("gzip -d " + filename + " -k").c_str());
  } else if (res == 3) {
    std::system(("unzip " + filename + " -d " + folderName).c_str());
  }

  if (MACOSX or UNIX or LINUX) {
    std::system(("rm " + filename).c_str());
  } else if (WIN) {
    std::system(("del " + filename).c_str());
  }

  return 1;
}

///
SuiteSparseDownloader::SuiteSparseDownloader(std::string matrixName,
                                             std::string location)
    : matrixName(matrixName), matrixLocation(location) {}

std::unordered_map<std::string, format::Format *>
SuiteSparseDownloader::download() {
  std::string downloadUrl;
  std::string matrixGroupName;
  std::string matrixName;
  std::string filepath;
  std::unordered_map<std::string, format::Format *> alternative_res;

  // Check if given argument is url or just group/matrixName
  std::regex regexURL(
      "\\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, "
      ".;]*[-a-zA-Z0-9+&@#/%=~_|])");
  if (std::regex_match(this->matrixName.c_str(), regexURL)) {
    downloadUrl = this->matrixName;
  } else {
    downloadUrl = "https://suitesparse-collection-website.herokuapp.com/MM/" +
                  this->matrixName + ".tar.gz";
  }

  // Extract matrix name and matrix group name to separate variables
  // Whether given argument is url or just group/matrixName
  int temp = 0;
  for (int i = downloadUrl.size() - 1, j = 0; i >= 0; i--) {
    if (downloadUrl[i] == '/') {
      if (j == 0) {
        filepath = downloadUrl.substr(i + 1, downloadUrl.size() - i - 1);
        temp = i;
        j = 1;
      } else {
        matrixGroupName = downloadUrl.substr(i + 1, temp - i - 1);
        break;
      }
    }
  }

  // Extracting matrix name from <matrixName>.tar.gz
  for (int i = 0; i < filepath.size(); i++) {
    if (filepath[i] == '.') {
      matrixName = filepath.substr(0, i);
      break;
    }
  }

  // Set the location for files.
  // If user did not give any specific location, then files will be located
  // in user's HOME directory.
  if (this->matrixLocation == "DEFAULT") {
    const char *homedir;

    if ((homedir = std::getenv("HOME")) == NULL) {
      homedir = getpwuid(getuid())->pw_dir;
    }
    std::filesystem::current_path(homedir);
  } else {
    std::filesystem::current_path(this->matrixLocation);
  }

  // SparseBase/ directory
  std::filesystem::path pForSparseBase("SparseBase");
  if (!(std::filesystem::exists(pForSparseBase))) {
    std::filesystem::create_directory(pForSparseBase);
  }
  std::filesystem::current_path(pForSparseBase);

  // SparseBase/cache directory
  std::filesystem::path pForCache("cache");
  if (!(std::filesystem::exists(pForCache))) {
    std::filesystem::create_directory(pForCache);
  }
  std::filesystem::current_path(pForCache);

  // SparseBase/cache/SuiteSparse directory
  std::filesystem::path pForSuiteSparse("SuiteSparse");
  if (!(std::filesystem::exists(pForSuiteSparse))) {
    std::filesystem::create_directory(pForSuiteSparse);
  }
  std::filesystem::current_path(pForSuiteSparse);

  // SparseBase/cache/SuiteSparse/<matrixGroup> directory
  std::filesystem::path pForMatrixGroup(matrixGroupName);
  if (!(std::filesystem::exists(pForMatrixGroup))) {
    std::filesystem::create_directory(pForMatrixGroup);
  }
  std::filesystem::current_path(pForMatrixGroup);

  // SparseBase/cache/SuiteSparse/<matrixGroup>/<matrixName> directory
  std::filesystem::path pForMatrix(matrixName);
  if (!(std::filesystem::exists(pForMatrix))) {
    if (Downloader::download(downloadUrl, filepath)) {
      int res = Extractor::extract(filepath);
      if (res == -1) {
        throw std::runtime_error("Error in unarchiving files!");
      }
    } else {
      std::cout << "Something went wrong!"
                << "\n";
    }
  }
  std::filesystem::current_path(pForMatrix);

  std::filesystem::path result_path(std::filesystem::current_path());

  for (auto &entry : std::filesystem::directory_iterator(result_path)) {
    std::string filename;
    std::string _path = entry.path().u8string();
    for (int i = _path.size(); i >= 0; i--) {
      if (_path[i] == '/') {
        filename = _path.substr(i + 1, _path.size() - i);
        break;
      }
    }

    filename = _path;
    std::ifstream fin(_path);
    if (fin.is_open()) {
      std::string line, buf;
      std::getline(fin, line);

      sparsebase::utils::MatrixMarket::MTXOptions options;
      options = sparsebase::utils::MatrixMarket::ParseHeader(line);

      if (options.field == 0) {
        sparsebase::io::MTXReader<ui, ui, float> reader(_path);

        if (options.format == 0) {
          format::COO<ui, ui, float> *coo = reader.ReadCOO();
          auto *_format = coo;
          alternative_res.insert({filename, _format});
        } else {
          format::Array<float> *arr = reader.ReadArray();
          format::Format *_format = arr;
          alternative_res.insert({filename, _format});
        }
      } else if (options.field == 1) {
        sparsebase::io::MTXReader<ui, ui, double> reader(_path);
        if (options.format == 0) {
          format::COO<ui, ui, double> *coo = reader.ReadCOO();
          format::Format *_format = coo;
          alternative_res.insert({filename, _format});
        } else {
          format::Array<double> *arr = reader.ReadArray();
          format::Format *_format = arr;
          alternative_res.insert({filename, _format});
        }
      } else if (options.field == 3) {
        sparsebase::io::MTXReader<ui, ui, int> reader(_path);
        if (options.format == 0) {
          format::COO<ui, ui, int> *coo = reader.ReadCOO();
          format::Format *_format = coo;
          alternative_res.insert({filename, _format});
        } else {
          format::Array<int> *arr = reader.ReadArray();
          format::Format *_format = arr;
          alternative_res.insert({filename, _format});
        }
      }
    }
    fin.close();
  }
  return alternative_res;
}

NetworkRepositoryDownloader::NetworkRepositoryDownloader(std::string matrixName,
                                                         std::string location)
    : matrixName(matrixName), matrixLocation(location) {}

void NetworkRepositoryDownloader::changePermission(std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    mode_t perm = st.st_mode;
    if (!(perm & S_IRUSR)) {
      chmod(path.c_str(), S_IRUSR);
      std::cout << "READ PERMISSION ENABLED!"
                << "\n";
    }
  }
}

std::unordered_map<std::string, format::Format *>
NetworkRepositoryDownloader::download() {
  std::string downloadUrl;
  std::string matrixGroupName;
  std::string matrixName;
  std::string filepath;
  std::unordered_map<std::string, format::Format *> alternative_res;

  // Check if given argument is url or just group/matrixName
  std::regex regexURL(
      "\\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, "
      ".;]*[-a-zA-Z0-9+&@#/%=~_|])");
  if (std::regex_match(this->matrixName.c_str(), regexURL)) {
    downloadUrl = this->matrixName;
  } else {
    //TODO:Construct links from a given matrix name, if possible
    throw std::runtime_error("Please provide a link!");
  }

  int temp = 0;
  for (int i = downloadUrl.size() - 1, j = 0; i >= 0; i--) {
    if (downloadUrl[i] == '/') {
      if (j == 0) {
        filepath = downloadUrl.substr(i + 1, downloadUrl.size() - i - 1);
        temp = i;
        j = 1;
      } else {
        matrixGroupName = downloadUrl.substr(i + 1, temp - i - 1);
        break;
      }
    }
  }

  // Extracting matrix name from <matrixName>.tar.gz
  for (int i = 0; i < filepath.size(); i++) {
    if (filepath[i] == '.') {
      matrixName = filepath.substr(0, i);
      break;
    }
  }

  // Set the location for files.
  // If user did not give any specific location, then files will be located
  // in user's HOME directory.
  if (this->matrixLocation == "DEFAULT") {
    const char *homedir;

    if ((homedir = std::getenv("HOME")) == NULL) {
      homedir = getpwuid(getuid())->pw_dir;
    }
    std::filesystem::current_path(homedir);
  } else {
    std::filesystem::current_path(this->matrixLocation);
  }

  // SparseBase/ directory
  std::filesystem::path pForSparseBase("SparseBase");
  if (!(std::filesystem::exists(pForSparseBase))) {
    std::filesystem::create_directory(pForSparseBase);
  }
  std::filesystem::current_path(pForSparseBase);

  // SparseBase/cache directory
  std::filesystem::path pForCache("cache");
  if (!(std::filesystem::exists(pForCache))) {
    std::filesystem::create_directory(pForCache);
  }
  std::filesystem::current_path(pForCache);

  // SparseBase/cache/SuiteSparse directory
  std::filesystem::path pForNetworkRepository("NetworkRepository");
  if (!(std::filesystem::exists(pForNetworkRepository))) {
    std::filesystem::create_directory(pForNetworkRepository);
  }
  std::filesystem::current_path(pForNetworkRepository);

  // SparseBase/cache/NetworkRepository/<matrixGroup> directory
  std::filesystem::path pForMatrixGroup(matrixGroupName);
  if (!(std::filesystem::exists(pForMatrixGroup))) {
    std::filesystem::create_directory(pForMatrixGroup);
  }
  std::filesystem::current_path(pForMatrixGroup);

  // SparseBase/cache/NetworkRepository/<matrixGroup>/<matrixName> directory
  std::filesystem::path pForMatrix(matrixName);
  if (!(std::filesystem::exists(pForMatrix))) {
    if (Downloader::download(downloadUrl, filepath)) {
      int res = Extractor::extract(filepath);
    } else {
      std::cout << "Something went wrong!"
                << "\n";
    }
  }
  std::filesystem::current_path(pForMatrix);

  std::filesystem::path result_path(std::filesystem::current_path());
  std::string result_path_string{result_path.u8string()};

  std::unordered_map<std::string, std::string> resulting_paths;

  for (auto &entry : std::filesystem::directory_iterator(result_path)) {
    std::string filename;
    std::string _path = entry.path().u8string();
    std::string fileExtension;
    for (int i = _path.size(); i >= 0; i--) {
      if (_path[i] == '/') {
        filename = _path.substr(i + 1, _path.size() - i);
        break;
      }
    }

    for (int i = filename.size(); i >= 0; i--) {
      if (filename[i] == '.') {
        fileExtension = filename.substr(i + 1, filename.size() - i);
        break;
      }
    }

    filename = _path;
    NetworkRepositoryDownloader::changePermission(_path);
    std::ifstream fin(_path);
    if (fin.is_open()) {
      if (fileExtension == "mtx") {
        std::string line, buf;
        std::getline(fin, line);

        sparsebase::utils::MatrixMarket::MTXOptions options;
        options = sparsebase::utils::MatrixMarket::ParseHeader(line);
        if (options.field == 0) {
          sparsebase::io::MTXReader<ui, ui, float> reader(_path);

          if (options.format == 0) {
            format::COO<ui, ui, float> *coo = reader.ReadCOO();
            auto *_format = coo;
            alternative_res.insert({filename, _format});
          } else {
            format::Array<float> *arr = reader.ReadArray();
            format::Format *_format = arr;
            alternative_res.insert({filename, _format});
          }
        } else if (options.field == 1) {
          sparsebase::io::MTXReader<ui, ui, double> reader(_path);
          if (options.format == 0) {
            format::COO<ui, ui, double> *coo = reader.ReadCOO();
            format::Format *_format = coo;
            alternative_res.insert({filename, _format});
          } else {
            format::Array<double> *arr = reader.ReadArray();
            format::Format *_format = arr;
            alternative_res.insert({filename, _format});
          }
        } else if (options.field == 3) {
          sparsebase::io::MTXReader<ui, ui, int> reader(_path);
          if (options.format == 0) {
            format::COO<ui, ui, int> *coo = reader.ReadCOO();
            format::Format *_format = coo;
            alternative_res.insert({filename, _format});
          } else {
            format::Array<int> *arr = reader.ReadArray();
            format::Format *_format = arr;
            alternative_res.insert({filename, _format});
          }
        }
      } else if (fileExtension == "edges") {
        bool weighted = false;
        std::string line, buf;
        std::getline(fin, line);
        std::istringstream stream(line);
        int counter = 0;
        while (stream >> buf) {
          counter++;
        }
        if (counter == 3) {
          weighted = true;
        }

        sparsebase::io::EdgeListReader<ui, ui, ui> edgeListReader(_path,
                                                                  weighted);
        format::CSR<ui, ui, ui> *csr = edgeListReader.ReadCSR();
        auto *_format = csr;
        alternative_res.insert({filename, _format});
      }
    }
    fin.close();
  }
  return alternative_res;
}

SnapDownloader::SnapDownloader(std::string matrixName, std::string location)
    : matrixName(matrixName), matrixLocation(location) {}

//TODO:Do not this method, instead add check to the EdgeListReader
void SnapDownloader::cleanFile(std::string path) {
  std::string filename = path;
  std::ifstream edgeFile(path);

  std::ofstream temp;
  temp.open("temp.txt");

  if (edgeFile.is_open()) {
    std::string line;
    while (std::getline(edgeFile, line)) {
      if (line[0] != '#') {
        temp << line << std::endl;
      }
    }
  }
  edgeFile.close();
  temp.close();
  std::rename("temp.txt", filename.c_str());
}

std::unordered_map<std::string, format::Format *> SnapDownloader::download() {
  std::string downloadUrl;
  std::string matrixGroupName;
  std::string matrixName;
  std::string filepath;
  std::unordered_map<std::string, format::Format *> alternative_res;

  // Check if given argument is url or just group/matrixName
  std::regex regexURL(
      "\\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, "
      ".;]*[-a-zA-Z0-9+&@#/%=~_|])");
  if (std::regex_match(this->matrixName.c_str(), regexURL)) {
    for (int i = this->matrixName.size(); i >= 0; i--) {
      if (this->matrixName[i] == '.') {
        if (this->matrixName.substr(i + 1, this->matrixName.size() - i - 1) ==
            "html") {
          downloadUrl = this->matrixName.substr(
                            0, this->matrixName.size() -
                                   (this->matrixName.size() - i - 1)) +
                        "txt.gz";
        } else {
          downloadUrl = this->matrixName;
        }
      }
    }
  } else {
    throw std::invalid_argument("SnapDownloader only excepts links!");
  }

  // Extract matrix name and matrix group name to separate variables
  // Whether given argument is url or just group/matrixName
  int temp = 0;
  for (int i = downloadUrl.size() - 1, j = 0; i >= 0; i--) {
    if (downloadUrl[i] == '/') {
      if (j == 0) {
        filepath = downloadUrl.substr(i + 1, downloadUrl.size() - i - 1);
        temp = i;
        j = 1;
      } else {
        matrixGroupName = downloadUrl.substr(i + 1, temp - i - 1);
        break;
      }
    }
  }

  // Extracting matrix name from <matrixName>.tar.gz
  for (int i = 0; i < filepath.size(); i++) {
    if (filepath[i] == '.') {
      matrixName = filepath.substr(0, i);
      break;
    }
  }

  // Set the location for files.
  // If user did not give any specific location, then files will be located
  // in user's HOME directory.
  if (this->matrixLocation == "DEFAULT") {
    const char *homedir;

    if ((homedir = std::getenv("HOME")) == NULL) {
      homedir = getpwuid(getuid())->pw_dir;
    }
    std::filesystem::current_path(homedir);
  } else {
    std::filesystem::current_path(this->matrixLocation);
  }

  // SparseBase/ directory
  std::filesystem::path pForSparseBase("SparseBase");
  if (!(std::filesystem::exists(pForSparseBase))) {
    std::filesystem::create_directory(pForSparseBase);
  }
  std::filesystem::current_path(pForSparseBase);

  // SparseBase/cache directory
  std::filesystem::path pForCache("cache");
  if (!(std::filesystem::exists(pForCache))) {
    std::filesystem::create_directory(pForCache);
  }
  std::filesystem::current_path(pForCache);

  // SparseBase/cache/SuiteSparse directory
  std::filesystem::path pForSnap("Snap");
  if (!(std::filesystem::exists(pForSnap))) {
    std::filesystem::create_directory(pForSnap);
  }
  std::filesystem::current_path(pForSnap);

  // SparseBase/cache/SuiteSparse/<matrixGroup> directory
  std::filesystem::path pForMatrixGroup(matrixGroupName);
  if (!(std::filesystem::exists(pForMatrixGroup))) {
    std::filesystem::create_directory(pForMatrixGroup);
  }
  std::filesystem::current_path(pForMatrixGroup);

  // SparseBase/cache/SuiteSparse/<matrixGroup>/<matrixName> directory
  std::filesystem::path pForMatrix(matrixName + ".txt");
  if (!(std::filesystem::exists(pForMatrix))) {
    if (Downloader::download(downloadUrl, filepath)) {
      int res = Extractor::extract(filepath);
      if (res == -1) {
        throw std::runtime_error("Error in unarchiving files!");
      }
    } else {
      throw std::invalid_argument("Something went wrong!");
    }
  }

  this->cleanFile(pForMatrix.string());

  sparsebase::io::EdgeListReader<ui, ui, ui> reader(pForMatrix.string());
  format::CSR<ui, ui, ui> *csr = reader.ReadCSR();
  auto *_format = csr;
  alternative_res.insert(
      {std::filesystem::absolute(pForMatrix).string(), _format});

  return alternative_res;
}
/*
FrosttDownloader::FrosttDownloader(std::string matrixName, std::string location)
    : matrixName(matrixName), matrixLocation(location) {}

std::unordered_map<std::string, format::Format *> FrosttDownloader::download() {
  std::string downloadUrl;
  std::string matrixGroupName;
  std::string matrixName;
  std::string filepath;
  std::unordered_map<std::string, format::Format *> alternative_res;

  // Check if given argument is url or just group/matrixName
  std::regex regexURL(
      "\\b((?:https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:, "
      ".;]*[-a-zA-Z0-9+&@#/%=~_|])");

  if (std::regex_match(this->matrixName.c_str(), regexURL)) {
    for (int i = this->matrixName.size(); i >= 0; i--) {
      if (this->matrixName[i] == '.') {
        if (this->matrixName.substr(i + 1, this->matrixName.size() - i - 1) ==
            "html") {
          downloadUrl = this->matrixName.substr(
                            0, this->matrixName.size() -
                                   (this->matrixName.size() - i - 1)) +
                        "txt.gz";
        } else {
          downloadUrl = this->matrixName;
        }
      }
    }
  } else {
    throw std::invalid_argument("Frostt only excepts links!");
  }

  // Extract matrix name and matrix group name to separate variables
  // Whether given argument is url or just group/matrixName
  int temp = 0;
  for (int i = downloadUrl.size() - 1, j = 0; i >= 0; i--) {
    if (downloadUrl[i] == '/') {
      if (j == 0) {
        filepath = downloadUrl.substr(i + 1, downloadUrl.size() - i - 1);
        temp = i;
        j = 1;
      } else {
        matrixGroupName = downloadUrl.substr(i + 1, temp - i - 1);
        std::cout << downloadUrl.substr(i + 1, temp - i - 1) << "\n";
        break;
      }
    }
  }

  // Extracting matrix name from <matrixName>.tar.gz
  for (int i = 0; i < filepath.size(); i++) {
    if (filepath[i] == '.') {
      matrixName = filepath.substr(0, i);
      break;
    }
  }

  // Set the location for files.
  // If user did not give any specific location, then files will be located
  // in user's HOME directory.
  if (this->matrixLocation == "DEFAULT") {
    const char *homedir;
    if ((homedir = std::getenv("HOME")) == NULL) {
      homedir = getpwuid(getuid())->pw_dir;
    }
    std::filesystem::current_path(homedir);
  } else {
    std::filesystem::current_path(this->matrixLocation);
  }

  // SparseBase/ directory
  std::filesystem::path pForSparseBase("SparseBase");
  if (!(std::filesystem::exists(pForSparseBase))) {
    std::filesystem::create_directory(pForSparseBase);
  }
  std::filesystem::current_path(pForSparseBase);

  // SparseBase/cache directory
  std::filesystem::path pForCache("cache");
  if (!(std::filesystem::exists(pForCache))) {
    std::filesystem::create_directory(pForCache);
  }
  std::filesystem::current_path(pForCache);

  // SparseBase/cache/Frostt directory
  std::filesystem::path pForSnap("Frostt");
  if (!(std::filesystem::exists(pForSnap))) {
    std::filesystem::create_directory(pForSnap);
  }
  std::filesystem::current_path(pForSnap);

  // SparseBase/cache/Frostt/<matrixGroup> directory
  std::filesystem::path pForMatrixGroup(matrixGroupName);
  if (!(std::filesystem::exists(pForMatrixGroup))) {
    std::filesystem::create_directory(pForMatrixGroup);
  }
  std::filesystem::current_path(pForMatrixGroup);

  // SparseBase/cache/SuiteSparse/<matrixGroup>/<matrixName> directory
  std::filesystem::path pForMatrix(matrixName);
  if (!(std::filesystem::exists(pForMatrix))) {
    if (Downloader::download(downloadUrl, filepath)) {
      int res = Extractor::extract(filepath);
      if (res == -1) {
        throw std::runtime_error("Error in unarchiving files!");
      }
    } else {
      throw std::invalid_argument("Something went wrong!");
    }
  }

  sparsebase::io::EdgeListReader<ui, ui, ui> reader(pathForData);
  format::CSR<ui, ui, ui> *csr = reader.ReadCSR();
  auto *_format = csr;
  alternative_res.insert({pathForData, _format});

  return alternative_res;
}
 */
}  // namespace io
}  // namespace sparsebase
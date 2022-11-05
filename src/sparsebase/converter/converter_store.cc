#include "converter_store.h"
#include <mutex>
namespace sparsebase::converter{
ConverterStore* ConverterStore::store_ = nullptr;
std::mutex ConverterStore::lock_;
}
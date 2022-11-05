#include "converter.h"
#include <unordered_map>
#include <typeindex>
#include <mutex>

#ifndef SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_STORE_H_
#define SPARSEBASE_SPARSEBASE_UTILS_CONVERTER_CONVERTER_STORE_H_
namespace sparsebase::converter {
class ConverterStore {
private:
    static ConverterStore* store_;
    static std::mutex lock_;
    std::unordered_map<std::type_index, std::weak_ptr<Converter>> type_to_wptr_;
    ConverterStore(){}
public:
    ConverterStore(const ConverterStore&) = delete;
    const ConverterStore& operator=(const ConverterStore&) = delete;
    static ConverterStore& GetStore(){
        lock_.lock();
        if (store_ == nullptr){
            store_ = new ConverterStore;
        }
        lock_.unlock();
        return *store_;
    }
    template <typename ConverterType>
    std::shared_ptr<Converter> get_converter() {
        lock_.lock();
        std::type_index type = typeid(ConverterType);
        std::shared_ptr<Converter> s_ptr;
        if (type_to_wptr_.find(type) == type_to_wptr_.end() || !(s_ptr = type_to_wptr_[type].lock())){
            s_ptr = std::make_shared<ConverterType>();
            type_to_wptr_[type] = s_ptr;
        }
        lock_.unlock();
        return std::static_pointer_cast<Converter>(s_ptr);
    }
};
}
#ifdef _HEADER_ONLY
#include "converter_store.cc"
#endif
#endif
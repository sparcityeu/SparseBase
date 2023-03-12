#include "sparsebase/io/patoh_reader.h"

#include <sstream>
#include <string>

#include "sparsebase/config.h"

namespace sparsebase::io{

template <typename IDType, typename NNZType, typename ValueType>
PatohReader<IDType, NNZType, ValueType>::PatohReader(std::string filename,
                                                     bool convert_to_zero_index)
        :filename_(filename),convert_to_zero_index_(convert_to_zero_index){
    std::ifstream fin(filename_);
    if(fin.is_open()){
        std::string header_line;
        std::getline(fin,header_line); // To skip the comment
        std::getline(fin,header_line); // Information about hypergraphs
        options_ = ParseHeader(header_line);
    }
    else{
        throw utils::ReaderException("Can not read HyperGraph\n");
    }
        }
template <typename IDType, typename NNZType, typename ValueType>
HyperGraphOptions PatohReader<IDType, NNZType, ValueType>::ParseHeader(
        std::string header_line) const {
    std::stringstream line_ss(header_line);
    HyperGraphOptions options;
    std::string base_type, cell_num, net_num, pin_num;
    line_ss >> base_type >> cell_num >> net_num >> pin_num;
    options.base_type = stoi(base_type);
    options.cell_num = stoi(cell_num);
    options.net_num = stoi(net_num);
    options.pin_num = stoi(pin_num);
    return options;
}           

}
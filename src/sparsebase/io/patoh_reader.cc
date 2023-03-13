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
template <typename IDType, typename NNZType, typename ValueType>
format::CSR<IDType, NNZType, ValueType>
    *PatohReader<IDType, NNZType, ValueType>::ReadCSR() const{
    
    std::ifstream fin(filename_);
    //Ignore headers and comments:
    int *pin_arr = new int[options_.pin_num];
    int *xpin_arr = new int[options_.net_num+1];
    int *netSize_arr = new int[options_.net_num]; // stores the vertex number of each net
    memset(netSize_arr, 0, options_.net_num * sizeof(int));
    std::string line;
    int i =0,k =0;
    while(std::getline(fin,line))
    {
        std::stringstream ss(line);
        if(line[0] != '%'){
            int num;
            int net_size = 0;
            while(ss>>num){
                pin_arr[i] = num;
                i++;
                net_size++;
            }
            netSize_arr[k] = net_size;
            k++;
        }
    }
    xpin_arr[0] = 0;
    for(int j=0; j<options_.net_num+1;j++)
    {
        xpin_arr[j+1] = xpin_arr[j]+netSize_arr[j];
    }

    int *row_ptr,col_ptr,*vals;

    row_ptr = (int *)malloc(sizeof(int)*(options_.net_num+1));
    memcpy(row_ptr,xpin_arr,sizeof(int) * (options_.net_num+1));

    
}

}
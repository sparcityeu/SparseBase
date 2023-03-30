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
        while (fin.peek() == '%')
            fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // To skip the comment
        std::getline(fin,header_line); // Information about hypergraphs
        options_ = ParseHeader(header_line);
    }
    else{
        throw utils::ReaderException("Can not read HyperGraph\n");
    }
        }
template <typename IDType, typename NNZType, typename ValueType>
typename PatohReader<IDType, NNZType, ValueType>::HyperGraphOptions 
PatohReader<IDType, NNZType, ValueType>::ParseHeader(
        std::string header_line) const {
    std::stringstream line_ss(header_line);
    HyperGraphOptions options;
    std::string base_type, cell_num, net_num, pin_num,weighted_scheme,constraint_num;
    line_ss >> base_type >> cell_num >> net_num >> pin_num>>weighted_scheme>>constraint_num;
    options.base_type = stoi(base_type);
    options.cell_num = stoi(cell_num);
    options.net_num = stoi(net_num);
    options.pin_num = stoi(pin_num);
    if (weighted_scheme != "") 
        options.weighted_scheme = stoi(weighted_scheme);
    if (constraint_num != "")
        options.constraint_num = stoi(constraint_num);
    return options;
}
template <typename IDType, typename NNZType, typename ValueType>

 void PatohReader<IDType, NNZType, ValueType>::ReadCSR(){
    
    std::ifstream fin(filename_);
    //Ignore headers and comments:
    int *pin_arr = new int[options_.pin_num];
    int *xpin_arr = new int[options_.net_num+1];
    xpin_arr[0] = 0;
    //Weight arrays of cells and nets
     int *cell_weight_arr = new int[options_.cell_num];
    std::fill(cell_weight_arr, cell_weight_arr + options_.cell_num, 1);
    int *net_weight_arr = new int[options_.net_num];
    std::fill(net_weight_arr, net_weight_arr + options_.net_num, 1);

    std::string line;
     while (fin.peek() == '%')
            fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // To skip the comment
    std::getline(fin, line); // To skip Information about hypergraphs
    int i =0,k =0,l=0;
     while (std::getline(fin, line))
    {
        std::stringstream ss(line);
        //If line is not comment
        if (line[0] != '%') {
            int num;
            int net_size = 0;
            if (options_.weighted_scheme == 2 || options_.weighted_scheme == 3) {
               if (!(options_.weighted_scheme == 3 && fin.eof())) {
                    ss >> num;
                    net_weight_arr[k] = num;
                }
            }
            while (ss >> num) {
                if((options_.weighted_scheme == 1 || options_.weighted_scheme == 3)&& (fin.eof()))// The last line of the file carries to information of cell's weights 
                {
                    cell_weight_arr[l] = num;
                    l++;
                }
                else {
                    pin_arr[i] = num;
                    i++;
                    net_size++;
                }
            }
            if(!((options_.weighted_scheme == 1 || options_.weighted_scheme == 3) && (fin.eof()))){
                xpin_arr[k + 1] = xpin_arr[k] + net_size;
                k++;
            }
        }
    }
    //int *row_ptr,col_ptr,*vals;
    //row_ptr = (int *)malloc(sizeof(int)*(options_.net_num+1));
    //memcpy(row_ptr,xpin_arr,sizeof(int) * (options_.net_num+1));


}

}
#include "sparsebase/io/patoh_reader.h"

#include <sstream>
#include <string>

#include "sparsebase/config.h"

namespace sparsebase::io{

template <typename IDType, typename NNZType, typename ValueType>
PatohReader<IDType, NNZType, ValueType>::PatohReader(std::string filename)
        :filename_(filename){
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
sparsebase::object::HyperGraph<IDType, NNZType, ValueType>
    *PatohReader<IDType, NNZType, ValueType>::ReadHyperGraph() const{
    
    std::ifstream fin(filename_);
    //Ignore headers and comments:
    IDType *pin_arr = new IDType[options_.pin_num];
    NNZType *xpin_arr = new NNZType[options_.net_num+1];
    IDType *net_arr = new IDType[options_.pin_num];
    NNZType *xnet_arr = new NNZType[options_.cell_num + 1];
    xpin_arr[0] = 0;
    xnet_arr[0] = 0;

    //Weight arrays of cells and nets
    //ValueType *cell_weight_arr = new ValueType[options_.cell_num];
    //std::fill(cell_weight_arr, cell_weight_arr + options_.cell_num, 1);
    //ValueType *net_weight_arr = new ValueType[options_.net_num];
    //std::fill(net_weight_arr, net_weight_arr + options_.net_num, 1);

    // Null arrays for filling the val array of CSR Matrix
    ValueType *xpin_val_arr = nullptr;
    ValueType *xnet_val_arr = nullptr;
    if constexpr (std::is_same_v<ValueType, void>) {
        std::string line;
        while (fin.peek() == '%')
                fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // To skip the comment
        std::getline(fin, line); // To skip Information about hypergraphs
        int i =0,k =0;
        while (std::getline(fin, line))
        {
            std::stringstream ss(line);
            //If line is not comment
            if (line[0] != '%') {
                int num;
                int net_size = 0;
                while (ss >> num) {
                        pin_arr[i] = num;
                        i++;
                        net_size++;
                }
                xpin_arr[k + 1] = xpin_arr[k] + net_size;
                k++;     
            }
        }

        if (options_.base_type == 0) {
            int add_index = 0;
            for (int m = 0; m < options_.cell_num; m++)
            {
                int vertice_size = 0;
                for (int n = 0; n < options_.pin_num; n++)
                {
                    if (m == pin_arr[n])
                    {
                        int temp_index = 0;
                        while (xpin_arr[temp_index + 1] < n + 1)
                        {
                            temp_index++;
                        }
                        net_arr[add_index] = temp_index;
                        add_index++;
                        vertice_size++;
                    }
                }
                xnet_arr[m + 1] = xnet_arr[m] + vertice_size;
            }
        }

        else if (options_.base_type == 1) { // shift m by 1
            int add_index = 0;
            for (int m = 0; m < options_.cell_num; m++)
            {
                int vertice_size = 0;
                for (int n = 0; n < options_.pin_num; n++)
                {
                    if (m+1 == pin_arr[n])
                    {
                        int temp_index = 0;
                        while (xpin_arr[temp_index + 1] < n + 1)
                        {
                            temp_index++;
                        }
                        net_arr[add_index] = temp_index+1;
                        add_index++;
                        vertice_size++;
                    }
                }
                xnet_arr[m + 1] = xnet_arr[m] + vertice_size;
            }
        }
    
        return new sparsebase::object::HyperGraph<IDType, NNZType, ValueType>(
            new format::CSR<IDType,NNZType,ValueType>(options_.net_num,options_.cell_num,xpin_arr,pin_arr,xpin_val_arr,sparsebase::format::kNotOwned,true),
            options_.base_type,
            options_.constraint_num,
            new format::CSR<IDType,NNZType,ValueType>(options_.cell_num,options_.net_num,xnet_arr,net_arr,xnet_val_arr,sparsebase::format::kNotOwned,true)

        );
    }

    else {
        ValueType *cell_weight_arr = new ValueType[options_.cell_num];
        std::fill(cell_weight_arr, cell_weight_arr + options_.cell_num, 1);
        ValueType *net_weight_arr = new ValueType[options_.net_num];
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

        if (options_.base_type == 0) {
            int add_index = 0;
            for (int m = 0; m < options_.cell_num; m++)
            {
                int vertice_size = 0;
                for (int n = 0; n < options_.pin_num; n++)
                {
                    if (m == pin_arr[n])
                    {
                        int temp_index = 0;
                        while (xpin_arr[temp_index + 1] < n + 1)
                        {
                            temp_index++;
                        }
                        net_arr[add_index] = temp_index;
                        add_index++;
                        vertice_size++;
                    }
                }
                xnet_arr[m + 1] = xnet_arr[m] + vertice_size;
            }
        }

        else if (options_.base_type == 1) { // shift m by 1
            int add_index = 0;
            for (int m = 0; m < options_.cell_num; m++)
            {
                int vertice_size = 0;
                for (int n = 0; n < options_.pin_num; n++)
                {
                    if (m+1 == pin_arr[n])
                    {
                        int temp_index = 0;
                        while (xpin_arr[temp_index + 1] < n + 1)
                        {
                            temp_index++;
                        }
                        net_arr[add_index] = temp_index+1;
                        add_index++;
                        vertice_size++;
                    }
                }
                xnet_arr[m + 1] = xnet_arr[m] + vertice_size;
            }
        }

        return new sparsebase::object::HyperGraph<IDType, NNZType, ValueType>(
            new format::CSR<IDType,NNZType,ValueType>(options_.net_num,options_.cell_num,xpin_arr,pin_arr,xpin_val_arr,sparsebase::format::kNotOwned,true),
            new format::Array<ValueType>(options_.net_num,net_weight_arr),
            new format::Array<ValueType>(options_.cell_num,cell_weight_arr),
            options_.base_type,
            options_.constraint_num,
            new format::CSR<IDType,NNZType,ValueType>(options_.cell_num,options_.net_num,xnet_arr,net_arr,xnet_val_arr,sparsebase::format::kNotOwned,true)
        );
    }
}
template <typename IDType, typename NNZType, typename ValueType>
PatohReader<IDType, NNZType, ValueType>::~PatohReader(){};
#ifndef _HEADER_ONLY
#include "init/patoh_reader.inc"
#endif
}
#include "sparsebase/io/patoh_writer.h"
#include <string>
#include "sparsebase/config.h"
#include "sparsebase/io/sparse_file_format.h"
#include "sparsebase/io/writer.h"
#include "sparsebase/format/csr.h"



namespace sparsebase::io {

template <typename IDType, typename NNZType, typename ValueType>
PatohWriter<IDType, NNZType, ValueType>::PatohWriter(
    std::string filename, bool is_zero_indexed, bool is_edge_weighted, bool is_vertex_weighted, int constraint_num) 
    : filename_(filename),
      is_zero_indexed_(is_zero_indexed),
      is_edge_weighted_(is_edge_weighted),
      is_vertex_weighted_(is_vertex_weighted) {}

template <typename IDType, typename NNZType, typename ValueType>
void PatohWriter<IDType, NNZType, ValueType>::WriteHyperGraph(
    object::HyperGraph<IDType, NNZType, ValueType> *hyperGraph) const {

    std::ofstream patohFile;
    patohFile.open(filename_);

    format::Format* con = hyperGraph->get_connectivity();
    auto base_type = hyperGraph->base_type_;
    auto constraint_num = hyperGraph->constraint_num_;
    auto cell_num = con->get_dimensions()[1]; // Cell Num = Vertices num
    auto net_num = hyperGraph->n_;
    auto pin_num = hyperGraph->m_;

    auto xpin_arr = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto pin_arr = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();

    auto xNetsCSR = hyperGraph->xNetCSR_;
    auto xnet_arr = xNetsCSR->get_row_ptr();
    auto net_arr = xNetsCSR->get_col();

    if constexpr (std::is_same_v<ValueType, void>){
        int index_num;
        if(is_zero_indexed_){
        index_num = 0;
        if(base_type == 1){ // If the hypergraph was 1 indexed convert it into 0 index
            for (int i = 0; i < pin_num; i++) {
                    pin_arr[i] -= 1;
                    net_arr[i] -= 1;
                } 
        }
        }

        else if(!is_zero_indexed_){
            index_num = 1;
            if(base_type == 0)
            {
                for (int i = 0; i < pin_num; i++) {
                    pin_arr[i] += 1;
                    net_arr[i] += 1;
                } 
            }
        }

        //Create Header Line
        std::string header_line;
        //header_line = std::to_string(index_num) + ' ' + std::to_string(cell_num) + ' ' + std::to_string(net_num) + ' ' + std::to_string(pin_num);
        header_line = std::to_string(index_num) + ' ' + std::to_string(cell_num) + ' ' + std::to_string(net_num) + ' ' + std::to_string(pin_num);
        patohFile << header_line;
        patohFile << "\n";
        int j = 0;
        int xpin_arr_size = 1;
        while (j < pin_num) {
                int xpin_arr_counter = 0;
                std::string str_line = "";
                while (xpin_arr_counter < (xpin_arr[xpin_arr_size] - xpin_arr[xpin_arr_size - 1])) {
                    str_line += std::to_string(pin_arr[j]) + " ";
                    j++;
                    xpin_arr_counter++;
                }
                str_line.erase(str_line.size() - 1);
                patohFile << str_line;
                if (j != pin_num)
                    patohFile << "\n";
                xpin_arr_size++;
            }
    }

    else{
        auto cell_weights = hyperGraph->cellWeights_;
        auto net_weights = hyperGraph->netWeights_;
        auto cell_weight_arr = cell_weights->get_vals();
        auto net_weight_arr = net_weights->get_vals();

        int index_num;
        int j = 0;
        int xpin_arr_size = 1;
        if(is_zero_indexed_){
        index_num = 0;
        if(base_type == 1){ // If the hypergraph was 1 indexed convert it into 0 index
            for (int i = 0; i < pin_num; i++) {
                    pin_arr[i] -= 1;
                    net_arr[i] -= 1;
                } 
        }
        }

        else if(!is_zero_indexed_){
            index_num = 1;
            if(base_type == 0)
            {
                for (int i = 0; i < pin_num; i++) {
                    pin_arr[i] += 1;
                    net_arr[i] += 1;
                } 
            }
        }

        if (!is_vertex_weighted_ && !is_edge_weighted_) {
            //Create Header Line
            std::string header_line;
            //int index_num = 0;
            header_line = std::to_string(index_num) + ' ' + std::to_string(cell_num) + ' ' + std::to_string(net_num) + ' ' + std::to_string(pin_num);
            patohFile << header_line;
            patohFile << "\n";

            while (j < pin_num) {
                int xpin_arr_counter = 0;
                std::string str_line = "";
                while (xpin_arr_counter < (xpin_arr[xpin_arr_size] - xpin_arr[xpin_arr_size - 1])) {
                    str_line += std::to_string(pin_arr[j]) + " ";
                    j++;
                    xpin_arr_counter++;
                }
                str_line.erase(str_line.size() - 1);
                patohFile << str_line;
                if (j != pin_num)
                    patohFile << "\n";
                xpin_arr_size++;

            }
        }

        else if (is_vertex_weighted_ && !is_edge_weighted_) {

            //Create Header Line
            std::string header_line;
            int weights_info = 1; // If only vertices are weighted

            header_line = std::to_string(index_num) + ' ' + std::to_string(cell_num) + ' ' + std::to_string(net_num) + ' ' + std::to_string(pin_num) + ' ' + std::to_string(weights_info);
            if(constraint_num != 1){ // No need to add the constraint num to the header if its value is 1 
                header_line += ' '+ std::to_string(constraint_num);
            }
            patohFile << header_line;
            patohFile << "\n";

            while (j < pin_num) {
                int xpin_arr_counter = 0;
                std::string str_line = "";
                while (xpin_arr_counter < (xpin_arr[xpin_arr_size] - xpin_arr[xpin_arr_size - 1])) {
                    str_line += std::to_string(pin_arr[j]) + " ";
                    j++;
                    xpin_arr_counter++;
                }
                str_line.erase(str_line.size() - 1);
                patohFile << str_line;
                patohFile << "\n";
                xpin_arr_size++;

            }

            // Insert cell weights
            int cell_weight_index = 0;
            std::string weight_line = "";
            while (cell_weight_index < cell_num) {
                weight_line += std::to_string(cell_weight_arr[cell_weight_index]) + " ";
                cell_weight_index++;
            }
            weight_line.erase(weight_line.size() - 1);
            patohFile << weight_line;
        }

        else if (!is_vertex_weighted_ && is_edge_weighted_) {
            //Create Header Line
            std::string header_line;
            int net_weights_index = 0;
            int weights_info = 2; // If only nets are weighted

            header_line = std::to_string(index_num) + ' ' + std::to_string(cell_num) + ' ' + std::to_string(net_num) + ' ' + std::to_string(pin_num) + ' ' + std::to_string(weights_info);
            if(constraint_num != 1){ // No need to add the constraint num to the header if its value is 1 
                header_line += ' '+ std::to_string(constraint_num);
            }
            patohFile << header_line;
            patohFile << "\n";

            while (j < pin_num) {
                int xpin_arr_counter = 0;
                std::string str_line = "";
                str_line += std::to_string(net_weight_arr[net_weights_index]) + " ";
                while (xpin_arr_counter < (xpin_arr[xpin_arr_size] - xpin_arr[xpin_arr_size - 1])) {
                    str_line += std::to_string(pin_arr[j]) + " ";
                    j++;
                    xpin_arr_counter++;
                }
                str_line.erase(str_line.size() - 1);
                patohFile << str_line;
                if (j != pin_num)
                    patohFile << "\n";
                xpin_arr_size++;
                net_weights_index++;

            }
        }

        else if (is_vertex_weighted_ && is_edge_weighted_) {
            //Create Header Line
            std::string header_line;
            //int index_num = 0;
            int net_weights_index = 0;
            int weights_info = 3; // Both Nets and Vertices are weighted
            header_line = std::to_string(index_num) + ' ' + std::to_string(cell_num) + ' ' + std::to_string(net_num) + ' ' + std::to_string(pin_num) + ' ' + std::to_string(weights_info);
            patohFile << header_line;
            patohFile << "\n";

            while (j < pin_num) {
                int xpin_arr_counter = 0;
                std::string str_line = "";
                str_line += std::to_string(net_weight_arr[net_weights_index]) + " ";
                while (xpin_arr_counter < (xpin_arr[xpin_arr_size] - xpin_arr[xpin_arr_size - 1])) {
                    str_line += std::to_string(pin_arr[j]) + " ";
                    j++;
                    xpin_arr_counter++;
                }
                str_line.erase(str_line.size() - 1);
                patohFile << str_line;
                patohFile << "\n";
                xpin_arr_size++;
                net_weights_index++;

            }

            // Insert cell weights
            int cell_weight_index = 0;
            std::string weight_line = "";
            while (cell_weight_index < cell_num) {
                weight_line += std::to_string(cell_weight_arr[cell_weight_index]) + " ";
                cell_weight_index++;
            }
            weight_line.erase(weight_line.size() - 1);
            patohFile << weight_line;

        }
    }
    patohFile.close();

}

#ifndef _HEADER_ONLY
#include "init/patoh_writer.inc"
#endif
}// namespace sparsebase::io
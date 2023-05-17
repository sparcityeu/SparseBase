#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <string>

#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/external/json/json.hpp"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
//#include "sparsebase/format/coo.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/io/mtx_reader.h"

template <typename IDType, typename NNZType, typename ValueType>
class Visualizer{

    public:
        Visualizer(sparsebase::format::CSR<IDType, NNZType, ValueType> *matrix,
                    nlohmann::json *feature_list, 
                    unsigned int bucket_size, 
                    bool plot_edges_by_weights){
            this->matrix = matrix;
            this->feature_list = feature_list;
            this->bucket_size = bucket_size;
            this->plot_edges_by_weights = plot_edges_by_weights;
            initHtml();
            plotNaturalOrdering();
            packHtml();
        };

        Visualizer(sparsebase::format::CSR<IDType, NNZType, ValueType> *matrix,
                    std::vector<sparsebase::reorder::Reorderer<IDType>*> *orderings,
                    nlohmann::json *feature_list,
                    unsigned int bucket_size,
                    bool plot_edges_by_weights){
            this->matrix = matrix;
            this->orderings = orderings;
            this->feature_list = feature_list;
            this->bucket_size = bucket_size;
            this->plot_edges_by_weights = plot_edges_by_weights;
            initHtml();
            plotNaturalOrdering();
            plotAlternateOrderings();
            packHtml();
        };
        
        std::string writeToHtml();

    private:
        void initHtml();
        void plotNaturalOrdering();
        void plotAlternateOrderings();
        void packHtml();

    private:
        sparsebase::format::CSR<IDType, NNZType, ValueType> *matrix; //
        std::vector<sparsebase::reorder::Reorderer<IDType>*> *orderings;
        nlohmann::json *feature_list;
        unsigned int bucket_size = 1;
        bool plot_edges_by_weights = false;
        std::string html = "";
        //std::vector<sparsebase::feature::FeaturePreprocessType<FeatureType>> *features;
};

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::initHtml() {
     html +=
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "  <head>\n"
            "    <meta charset=\"UTF-8\" />\n"
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
            "    <meta http-equiv=\"X-UA-Compatible\" content=\"ie=edge\" />\n"
            "    <title>Visualization</title>\n"
            "    <link rel=\"stylesheet\" href=\"style.css\" />\n"
            "    <script src=\"https://cdnjs.cloudflare.com/ajax/libs/jquery/2.2.0/jquery.min.js\"></script>\n"
            "  </head>\n"
            "  <body>\n"
            "    <div class=\"header\">\n"
            "      <h1>Name of Matrix/Graph</h1>\n"
            "    </div>\n"
            "    <div class=\"content\">\n"
            "      <div class=\"non-ordering-based-features\">";
            for (const auto& feature : (*feature_list)["non_ordering_based_features"].items())
            {
                html += "<div class=\"card\">\n"
                        "  <h3>"+feature.key()+"</h3>\n"
                        "  <p>" +feature.value().dump()+"</p>\n"
                        "</div>";
            }

     html+= "</div>\n";
}

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::plotNaturalOrdering() {
    //auto *coo = toCoo(matrix);
    //int bucketSize = 22000000;//190000;
    auto x_dim = ceil(matrix->get_dimensions()[0] / (double)bucket_size);
    auto y_dim = ceil(matrix->get_dimensions()[1] / (double)bucket_size);
    std::vector<std::vector<long long>> resulting_matrix(x_dim, std::vector<long long>(y_dim, 0));
    auto r = matrix->get_row_ptr();
    auto c = matrix->get_col();
    long long tmp_elem_index = 0;


    for (int i = 0; i < matrix->get_dimensions()[0]; i++) {
        for(int j=r[i]; j<r[i+1]; j++){
            //int k = (*orderings)[0]
            int k = i / (double)bucket_size;
            int l = c[j] / (double)bucket_size;
            if(l>=y_dim){
                std::cout << "r: " << i << " c: " << c[j] << " k " << k << " l " << l << "\n";
                std::cout << "ydim: " << y_dim << " l: " << l << std::endl;
            }
            assert(k < x_dim);
            assert(l < y_dim);
            assert(k>-1);
            assert(l>-1);
            if(plot_edges_by_weights)
                resulting_matrix[k][l] += abs(matrix->get_vals()[tmp_elem_index++]);
            else
                resulting_matrix[k][l]++;
        }
    }

    std::string order_name = (*feature_list)["features_list"][0]["order_name"];

    html+=  "<div class=\"ordering-based-features\">\n"
            "  <div class=\"section\">\n"
            "    <div class=\"left-section\">\n"
            "      <h2>"+order_name+"</h2>\n"
            "          <div id=\"plot0\" class=\"matrix\"></div>\n" // plot0
            "    </div>\n"
            "    <div class=\"middle-section\">\n"
            "      <div class=\"info-box\">\n"
            "        <h3>Information</h3>\n"
            "        <p>lorem ipsum</p>\n"
            "        <p>lorem ipsum</p>\n"
            "        <p>lorem ipsum</p>\n"
            "      </div>\n"
            "      <div class=\"feature-box\">\n"
            "        <h3>Ordering Based Features</h3>\n"
            "        <ul>";

    for (const auto& feature : (*feature_list)["features_list"][0]["features"].items())
    {
        html+=  "<li>"+feature.key()+": "+feature.value().dump()+"</li>\n";
    }

    html+=  "     </ul>\n"
            "    </div>\n"
            "  </div>\n"
            "  <div class=\"right-section\">\n"
            "    <div class=\"graphical-box\">\n"
            "      <h3>Graphical Features</h3>\n"
            "      <div class=\"graph\"><p>insert graph here</p></div>\n"
            "      <div class=\"graph\"><p>insert graph here</p></div>\n"
            "    </div>\n"
            "  </div>\n"
            "</div>";
    
    html+=  "<div class=\"footer\"></div>\n"
            "<script src=\"https://cdn.plot.ly/plotly-1.45.3.min.js\"></script>\n"
            "<script>\n"
            "  function maximizar()\n"
            "  {Plotly.relayout(document.getElementById('plot0'));}\n"
            "  ;window.onresize = function() {$('.js-plotly-plot')\n"
            "  .each(function(index, gd) {Plotly.Plots.resize(gd)\n"
            "    .then(function(){maximizar()});});};Plotly\n"
            "    .react(document.getElementById('plot0'),\n"
            "    [";

    nlohmann::json json_plot;
    json_plot["type"] = "heatmap";
    json_plot["z"] = resulting_matrix;
    json_plot["xgap"] = 1;
    json_plot["ygap"] = 1;
    json_plot["colorscale"] = "YlOrRd";
    json_plot["reversescale"] = true;
    json_plot["hovertemplate"] =
        "X: %{x}<br>Y: %{y}<br>NNZ(s): %{z}<extra></extra>";

    html += json_plot.dump();
    html +=
        "], {yaxis: {dtick: 1, \"autorange\": \"reversed\"}, xaxis: {dtick: "
        "1, \"side\":\"top\"}, zaxis: {title:\"NNZ(s)\"}, plot_bgcolor:\"rgba(0,0,0,1)\", paper_bgcolor:\"rgba(0,0,0,0)\"}";  // use json
                                                                // dump of the
                                                                // layout;
    html += "); \nmaximizar();";
    html += "</script> \n";
    // std::cout << html;
}

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::plotAlternateOrderings() {
    
    for(int orderIndex = 0; orderIndex < (*orderings).size(); orderIndex++){
        sparsebase::context::CPUContext cpu_context;
        sparsebase::format::FormatOrderTwo<IDType , NNZType , ValueType >* reordered = reorder_custom<IDType, NNZType, ValueType>(matrix, 
            (*orderings)[orderIndex], {&cpu_context}, true);
        
        sparsebase::format::CSR<IDType , NNZType , ValueType >* csr = static_cast<sparsebase::format::CSR<IDType , NNZType , ValueType>*>(reordered);
        //int bucketSize = 22000000;
        auto x_dim = ceil(csr->get_dimensions()[0] / (double)bucket_size);
        auto y_dim = ceil(csr->get_dimensions()[1] / (double)bucket_size);
        auto r = csr->get_row_ptr();
        auto c = csr->get_col();
        std::vector<std::vector<long long>> resulting_matrix(x_dim, std::vector<long long>(y_dim, 0));

        long long tmp_elem_index = 0;
        //std::cout << "bucketSize: " << bucketSize << " dim0: " << csr->get_dimensions()[0] << " dim1: " << csr->get_dimensions()[1] << "\n";
        for (int i = 0; i < csr->get_dimensions()[0]; i++) {
            for(int j=r[i]; j<r[i+1]; j++){
                //int k = (*orderings)[0]
                int k = i / (double)bucket_size;
                int l = c[j] / (double)bucket_size;
                if(l>=y_dim){
                    std::cout << "r: " << i << " c: " << c[j] << " k " << k << " l " << l << "\n";
                    std::cout << "ydim: " << y_dim << " l: " << l << std::endl;
                }
                assert(k < x_dim);
                assert(l < y_dim);
                assert(k>-1);
                assert(l>-1);
                if(plot_edges_by_weights)
                    resulting_matrix[k][l] += abs(csr->get_vals()[tmp_elem_index++]);
                else
                    resulting_matrix[k][l]++;
            }
        }

        std::string order_name = (*feature_list)["features_list"][orderIndex+1]["order_name"];

       html += "<div class=\"section\">\n"
               "  <div class=\"left-section\">\n"
               "    <h2>Degree Reorder</h2>\n"
               "      <div id=\"plot1\" class=\"matrix\"></div>\n"
               "  </div>\n"
               "  <div class=\"middle-section\">\n"
               "    <div class=\"info-box\">\n"
               "      <h3>Information</h3>\n"
               "      <p>lorem ipsum</p>\n"
               "      <p>lorem ipsum</p>\n"
               "      <p>lorem ipsum</p>\n"
               "    </div>\n"
               "    <div class=\"feature-box\">\n"
               "      <h3>Ordering Based Features</h3>\n"
               "      <ul>";

        for (const auto& feature : (*feature_list)["features_list"][orderIndex]["features"].items())
        {
            html+= "<li>" + feature.key() + ": " + feature.value().dump() + "</li>";
        }
        
        html+=
        "      </ul>\n"
        "    </div>\n"
        "  </div>\n"
        "  <div class=\"right-section\">\n"
        "    <div class=\"graphical-box\">\n"
        "      <h3>Graphical Features</h3>\n"
        "      <div class=\"graph\"><p>insert graph here</p></div>\n"
        "      <div class=\"graph\"><p>insert graph here</p></div>\n"
        "    </div>\n"
        "  </div>\n"
        "</div> "
        "<script "
        "src=\"https://cdn.plot.ly/plotly-1.45.3.min.js\"></script>\n"
        "<script>\n"
        "function "
        "maximizar(){Plotly.relayout(document.getElementById('plot" + std::to_string(orderIndex+1) +"')"
        ");};window.onresize = function() "
        "{$('.js-plotly-plot').each(function(index, gd) "
        "{Plotly.Plots.resize(gd).then(function(){maximizar()});});};"
        "Plotly.react(document.getElementById('plot"+ std::to_string(orderIndex+1)+"'),[";
        
        nlohmann::json json_plot;
        json_plot["type"] = "heatmap";
        json_plot["z"] = resulting_matrix;
        json_plot["xgap"] = 1;
        json_plot["ygap"] = 1;
        json_plot["colorscale"] = "YlOrRd"; //"Greys";
        json_plot["reversescale"] = true;
        json_plot["hovertemplate"] =
            "X: %{x}<br>Y: %{y}<br>NNZ(s): %{z}<extra></extra>";

        html += json_plot.dump();
        html += "], ";
        html +=
            "{yaxis: {dtick: 1, \"autorange\": \"reversed\"}, xaxis: {dtick: "
            "1, \"side\":\"top\"}, zaxis: {title:\"NNZ(s)\"}, plot_bgcolor:\"rgba(0,0,0,1)\", paper_bgcolor:\"rgba(0,0,0,0)\"}";  // use json
                                                                    // dump of the
                                                                    // layout;
        html += "); \nmaximizar();";
        html += "</script> \n";
    }
    html+= "  </div>\n"
           "</div>";
}

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::packHtml() {
    html += "</body> \n </html> \n";
}

template <typename IDType, typename NNZType, typename ValueType>
std::string Visualizer<IDType, NNZType, ValueType>::writeToHtml() {
     return html;
}

int main(void) {
    /*
    int csr_row_ptr[5]{0, 2, 3, 3, 4};
    int csr_col[4]{0, 2, 1, 3};
    int csr_vals[4]{4, 5, 7, 9};
    sparsebase::format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col, csr_vals);
    */

    std::string file_name = "mawi_201512020330.mtx";
    sparsebase::io::MTXReader<unsigned int, unsigned int, float> reader(file_name);
    sparsebase::format::CSR<unsigned int, unsigned int, float> *csr =
        reader.ReadCSR();

    sparsebase::reorder::DegreeReorder<unsigned int,unsigned int,float> orderer(1);
    std::vector<sparsebase::reorder::Reorderer<unsigned int>*> orderings;
    orderings.push_back(&orderer);

    nlohmann::json featuresList = {
        {"non_ordering_based_features", {
            {"number_triangles", 0},         // unsigned integer
            {"max_degree", 0},               // unsigned integer
            {"avg_degree", 0.0},             // float ?
            {"betweenness_centrality", 0.0}, // float
            {"symmetry_ratio", 0.0}          // float
        }},
        {"features_list", {
            {
                {"order_name", "Natural Ordering"},
                {"features", {
                    {"number_diagonal_entries", 0},            // unsigned integer
                    {"number_non_diagonal_entries", 0},        // unsigned integer
                    {"number_dense_rows", 0},                  // unsigned integer
                    {"bandwidth", 0.0}                         // float ?

                }}
            },
            {
                {"order_name", "Degree Reorder"},
                {"features", {
                    {"number_diagonal_entries", 0},            // unsigned integer
                    {"number_non_diagonal_entries", 0},        // unsigned integer
                    {"number_dense_rows", 0},                  // unsigned integer
                    {"bandwidth", 0.0}                         // float ?
                }}
            }
        }}
    };

    unsigned int bucketSize = 22000000;
    bool doPlotByWeights = true;
    Visualizer<unsigned int,unsigned int,float>vsl (csr, &orderings, &featuresList, bucketSize, doPlotByWeights);
    //std::cout << vsl.writeToHtml();

    std::ofstream file("myfile.html");
    if (file.is_open()) {
        file << vsl.writeToHtml();
        file.close();
    }
    else {
        std::cout << "Error: Unable to open file!" << std::endl;
    }
                
    return 0;
}
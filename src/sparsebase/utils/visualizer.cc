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
#include "sparsebase/format/coo.h" //
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
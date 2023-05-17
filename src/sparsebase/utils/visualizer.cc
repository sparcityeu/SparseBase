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
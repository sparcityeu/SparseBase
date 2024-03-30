#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/context/cpu_context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/external/json/json.hpp"
#include "sparsebase/feature/feature_preprocess_type.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/io/mtx_reader.h"
#include "sparsebase/reorder/reorderer.h"
// #include "sparsebase/format/coo.h"

template <typename IDType, typename NNZType, typename ValueType,
          typename FeatureType>
sparsebase::format::FormatOrderTwo<IDType, NNZType, FeatureType>*
reorder_custom(
    sparsebase::format::FormatOrderTwo<IDType, NNZType, FeatureType>* input,
    sparsebase::reorder::Reorderer<IDType>* reordering,
    std::vector<sparsebase::context::Context*> contexts, bool convert_inputs) {
  // Summary: Converts the input FormatOrderTwo form to a specified format.
  // Returns: The reordered form of the input.
  IDType* perm_vector = reordering->GetReorder(input, contexts, convert_inputs);
  return sparsebase::bases::ReorderBase::Permute2D(perm_vector, input, contexts,
                                                   convert_inputs);
};

template <typename IDType, typename NNZType, typename ValueType>
class Visualizer {
 public:
  // Summary: Constructor that can be used if only natural order form of the
  // matrix is to be plotted. Params: Takes a pointer to the CSR matrix, a
  // feature_list json object, bucket_size unsigned int and do_plot_by_weights
  // flag as parameters.
  Visualizer(sparsebase::format::CSR<IDType, NNZType, ValueType>* matrix,
             nlohmann::json* feature_list, unsigned int bucket_size,
             bool plot_edges_by_weights) {
    this->matrix = matrix;
    this->feature_list = feature_list;
    this->bucket_size = bucket_size;
    this->plot_edges_by_weights = plot_edges_by_weights;
    initHtml();
    plotNaturalOrdering();
    packHtml();
  };

  // Summary: Constructor that can be used if only natural order form of the
  // matrix is to be plotted. Params: Takes a pointer to the CSR matrix, a
  // vector of desired alternate orderings to be plotted, a feature_list json
  // object, bucket_size unsigned int and do_plot_by_weights flag as parameters.
  // Note: Example usage of the constructor can be found in the main function at
  // the bottom.
  Visualizer(sparsebase::format::CSR<IDType, NNZType, ValueType>* matrix,
             std::vector<sparsebase::reorder::Reorderer<IDType>*>* orderings,
             nlohmann::json* feature_list, unsigned int bucket_size,
             bool plot_edges_by_weights) {
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

  std::string getHtml();  // Getter for the html string

 private:
  void initHtml();
  void plotNaturalOrdering();
  void plotAlternateOrderings();
  void packHtml();

 private:
  sparsebase::format::CSR<IDType, NNZType, ValueType>* matrix;
  std::vector<sparsebase::reorder::Reorderer<IDType>*>* orderings;
  nlohmann::json* feature_list;
  unsigned int bucket_size = 1;
  bool plot_edges_by_weights = false;
  std::string html = "";
};

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::initHtml() {
  html +=
      // Initiliaze the html
      "<!DOCTYPE html>\n"
      "<html lang=\"en\">\n"
      "  <head>\n"
      "    <meta charset=\"UTF-8\" />\n"
      "    <meta name=\"viewport\" content=\"width=device-width, "
      "initial-scale=1.0\" />\n"
      "    <meta http-equiv=\"X-UA-Compatible\" content=\"ie=edge\" />\n"
      "    <title>Visualization</title>\n"

      // Get the styles from styles.css
      "    <link rel=\"stylesheet\" href=\"style.css\" />\n"

      "    <script "
      "src=\"https://cdnjs.cloudflare.com/ajax/libs/jquery/2.2.0/"
      "jquery.min.js\"></script>\n"
      "  </head>\n"
      "  <body>\n"
      "    <div class=\"header\">\n"

      // Display the name of the matrix/graph on html.
      "      <h1>Name of Matrix/Graph</h1>\n"
      "    </div>\n"
      "    <div class=\"content\">\n"

      // Non-orderings based features are displayed on the top section
      "      <div class=\"non-ordering-based-features\">";

  // Add the key, values of non-ordering based features from json object
  for (const auto& feature :
       (*feature_list)["non_ordering_based_features"].items()) {
    html +=
        "<div class=\"card\">\n"
        "  <h3>" +
        feature.key() +
        "</h3>\n"
        "  <p>" +
        feature.value().dump() +
        "</p>\n"
        "</div>";
  }

  html += "</div>\n";
}

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::plotNaturalOrdering() {
  // Summary: Generates the html corresponding to the natural order plot and its
  // features
  //          Appends the generated html to the overall html string.

  // Decide on the plot dimensions based on original matrix dimensions and
  // specified bucket size
  auto x_dim = ceil(matrix->get_dimensions()[0] / (double)bucket_size);
  auto y_dim = ceil(matrix->get_dimensions()[1] / (double)bucket_size);

  // Define a 2D vector for the plot, set row pointer and col.
  std::vector<std::vector<long long>> resulting_matrix(
      x_dim, std::vector<long long>(y_dim, 0));
  auto r = matrix->get_row_ptr();
  auto c = matrix->get_col();
  long long tmp_elem_index = 0;

  // Iterate through the original matrix
  for (int i = 0; i < matrix->get_dimensions()[0]; i++) {
    for (int j = r[i]; j < r[i + 1]; j++) {
      // Calculate the corresponding cell coordinates on the plot matrix.
      // Converts from original matrix' cell coordinates to plot matrix' cell
      // coordinates using bucket_size e.g. 15x15 matrix is plotted as 3x3. Cell
      // at (8,12) index will correspond to (1,2) on the plot matrix
      int k = i / (double)bucket_size;
      int l = c[j] / (double)bucket_size;

      // Check for out-of-bound indices
      assert(k < x_dim);
      assert(l < y_dim);
      assert(k > -1);
      assert(l > -1);

      // If plot_edges_by_weights flag is "true", add its absolute value to its
      // corresponding cell on the plot If set to false, will plot based on
      // frequency, and increment cell value by 1
      if (plot_edges_by_weights)
        resulting_matrix[k][l] += abs(matrix->get_vals()[tmp_elem_index++]);
      else
        resulting_matrix[k][l]++;
    }
  }

  // Get the name of the ordering from the features_list json object to be
  // printed on the html.
  std::string order_name = (*feature_list)["features_list"][0]["order_name"];

  // Add the html sections for the plot and its features
  html +=
      "<div class=\"ordering-based-features\">\n"
      "  <div class=\"section\">\n"
      // Left section
      "    <div class=\"left-section\">\n"
      "      <h2>" +
      order_name +
      "</h2>\n"
      "          <div id=\"plot0\" class=\"matrix\"></div>\n"  // plot0
      "    </div>\n"

      // Middle section
      "    <div class=\"middle-section\">\n"
      "      <div class=\"feature-box\">\n"
      "        <h3>Ordering Based Features</h3>\n"
      "        <ul>";

  // Add the features and their values from the json object
  for (const auto& feature :
       (*feature_list)["features_list"][0]["features"].items()) {
    html += "<li>" + feature.key() + ": " + feature.value().dump() + "</li>\n";
  }

  html +=
      "     </ul>\n"
      "    </div>\n"
      "  </div>\n"

      // End of the middle section, start of the right section
      "  <div class=\"right-section\">\n"
      "    <div class=\"graphical-box\">\n"
      "      <h3>Graphical Features</h3>\n"
      "      <div class=\"graph\"><p>insert graph here</p></div>\n"
      "      <div class=\"graph\"><p>insert graph here</p></div>\n"
      "    </div>\n"
      "  </div>\n"
      "</div>";

  // Add the plot.ly script to the HTML string
  html +=
      "<div class=\"footer\"></div>\n"
      "<script src=\"https://cdn.plot.ly/plotly-1.45.3.min.js\"></script>\n"
      "<script>\n"
      "  function maximizar()\n"
      "  {Plotly.relayout(document.getElementById('plot0'));}\n"
      "  ;window.onresize = function() {$('.js-plotly-plot')\n"
      "  .each(function(index, gd) {Plotly.Plots.resize(gd)\n"
      "    .then(function(){maximizar()});});};Plotly\n"
      "    .react(document.getElementById('plot0'),\n"
      "    [";

  // Set plotly heatmap parameters by forming a json object
  nlohmann::json json_plot;
  json_plot["type"] = "heatmap";
  json_plot["z"] = resulting_matrix;
  json_plot["xgap"] = 1;  // xgap and ygap adds the small black border effect
                          // for the cell on the graph
  json_plot["ygap"] = 1;
  json_plot["colorscale"] = "YlOrRd";
  json_plot["reversescale"] = true;
  json_plot["hovertemplate"] =
      "X: %{x}<br>Y: %{y}<br>NNZ(s): %{z}<extra></extra>";

  // dump the graph parameters to the html string
  html += json_plot.dump();

  // Generate annotations to display "X" symbols on empty plot cells.
  std::string annotationsStr = "[";
  for (int i = 0; i < x_dim; i++) {
    for (int j = 0; j < y_dim; j++) {
      if (resulting_matrix[i][j] == 0) {
        // Generate a single annotation object
        std::string annotation = "{x: " + std::to_string(i) +
                                 ", y: " + std::to_string(j) +
                                 ", text: 'X', showarrow: false}, ";
        annotationsStr += annotation;
      }
    }
  }
  // Remove the trailing comma and space
  if (annotationsStr.length() > 1) {
    annotationsStr = annotationsStr.substr(0, annotationsStr.length() - 2);
  }
  annotationsStr += "]";

  // If x axis cell count is larger than set threshold (25), do not display tick
  // labels to prevent distorted graphs
  nlohmann::json json_xaxis;
  json_xaxis["dtick"] = 1;
  json_xaxis["side"] = "top";
  if (x_dim > 25) {
    json_xaxis["showticklabels"] = false;
  }

  // If y axis cell count is larger than set threshold (25), do not display tick
  // labels to prevent distorted graphs
  nlohmann::json json_yaxis;
  json_yaxis["dtick"] = 1;
  json_yaxis["autorange"] = "reversed";
  if (y_dim > 25) {
    json_yaxis["showticklabels"] = false;
  }

  // Set the layout parameters of the graph.
  // (Might be better to convert it to a json object or another format if
  // further modifications are needed.)
  html += std::string("], {yaxis: ") + json_yaxis.dump() +
          ", xaxis: " + json_xaxis.dump() +
          ", zaxis: {title:\"NNZ(s)\"}, plot_bgcolor:\"rgba(0,0,0,1)\", "
          "paper_bgcolor:\"rgba(0,0,0,0)\", annotations: " +
          annotationsStr;
  html += "}); \nmaximizar();";
  html += "</script> \n";
}

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::plotAlternateOrderings() {
  // Iterate through all the desired orderings
  for (int orderIndex = 0; orderIndex < (*orderings).size(); orderIndex++) {
    // Convert the original matrix to the desired ordering.
    // Note: This way of conversion may be incorrect, or there might be a bug
    // with the converter. Check later.
    sparsebase::context::CPUContext cpu_context;
    sparsebase::format::FormatOrderTwo<IDType, NNZType, ValueType>* reordered =
        reorder_custom<IDType, NNZType, ValueType>(
            matrix, (*orderings)[orderIndex], {&cpu_context}, true);
    sparsebase::format::CSR<IDType, NNZType, ValueType>* csr =
        static_cast<sparsebase::format::CSR<IDType, NNZType, ValueType>*>(
            reordered);

    // Decide on the plot dimensions based on original matrix dimensions and
    // specified bucket size
    auto x_dim = ceil(csr->get_dimensions()[0] / (double)bucket_size);
    auto y_dim = ceil(csr->get_dimensions()[1] / (double)bucket_size);

    // Define a 2D vector for the plot, set row pointer and col.
    auto r = csr->get_row_ptr();
    auto c = csr->get_col();
    std::vector<std::vector<long long>> resulting_matrix(
        x_dim, std::vector<long long>(y_dim, 0));
    long long tmp_elem_index = 0;

    // Iterate through the original matrix
    for (int i = 0; i < csr->get_dimensions()[0]; i++) {
      for (int j = r[i]; j < r[i + 1]; j++) {
        // Calculate the corresponding cell coordinates on the plot matrix.
        // Converts from original matrix' cell coordinates to plot matrix' cell
        // coordinates using bucket_size e.g. 15x15 matrix is plotted as 3x3.
        // Cell at (8,12) index will correspond to (1,2) on the plot matrix
        int k = i / (double)bucket_size;
        int l = c[j] / (double)bucket_size;

        // Check for out-of-bound indices
        assert(k < x_dim);
        assert(l < y_dim);
        assert(k > -1);
        assert(l > -1);

        // If plot_edges_by_weights flag is "true", add its absolute value to
        // its corresponding cell on the plot If set to false, will plot based
        // on frequency, and increment cell value by 1
        if (plot_edges_by_weights)
          resulting_matrix[k][l] += abs(csr->get_vals()[tmp_elem_index++]);
        else
          resulting_matrix[k][l]++;
      }
    }

    // Get the name of the ordering from the features_list json object to be
    // printed on the html.
    std::string order_name =
        (*feature_list)["features_list"][orderIndex + 1]["order_name"];

    // Add the html sections for the plot and its features
    html +=
        "<div class=\"section\">\n"
        // Left section
        "  <div class=\"left-section\">\n"
        "    <h2>Degree Reorder</h2>\n"  // MODIFY
        "      <div id=\"plot1\" class=\"matrix\"></div>\n"
        "  </div>\n"

        // Middle section
        "  <div class=\"middle-section\">\n"
        "    <div class=\"feature-box\">\n"
        "      <h3>Ordering Based Features</h3>\n"
        "      <ul>";

    // Add the features and their values from the json object
    for (const auto& feature :
         (*feature_list)["features_list"][orderIndex]["features"].items()) {
      html += "<li>" + feature.key() + ": " + feature.value().dump() + "</li>";
    }

    html +=
        "      </ul>\n"
        "    </div>\n"
        "  </div>\n"

        // End of the middle section, start of the right section
        "  <div class=\"right-section\">\n"
        "    <div class=\"graphical-box\">\n"
        "      <h3>Graphical Features</h3>\n"
        "      <div class=\"graph\"><p>insert graph here</p></div>\n"
        "      <div class=\"graph\"><p>insert graph here</p></div>\n"
        "    </div>\n"
        "  </div>\n"
        "</div> "

        // Add the plot.ly script to the HTML string
        "<script "
        "src=\"https://cdn.plot.ly/plotly-1.45.3.min.js\"></script>\n"
        "<script>\n"
        "function "
        "maximizar(){Plotly.relayout(document.getElementById('plot" +
        std::to_string(orderIndex + 1) +
        "')"
        ");};window.onresize = function() "
        "{$('.js-plotly-plot').each(function(index, gd) "
        "{Plotly.Plots.resize(gd).then(function(){maximizar()});});};"
        "Plotly.react(document.getElementById('plot" +
        std::to_string(orderIndex + 1) + "'),[";

    // Set plotly heatmap parameters by forming a json object
    nlohmann::json json_plot;
    json_plot["type"] = "heatmap";
    json_plot["z"] = resulting_matrix;
    json_plot["xgap"] = 1;  // xgap and ygap adds the small black border effect
                            // for the cell on the graph
    json_plot["ygap"] = 1;
    json_plot["colorscale"] = "YlOrRd";
    json_plot["reversescale"] = true;
    json_plot["hovertemplate"] =
        "X: %{x}<br>Y: %{y}<br>NNZ(s): %{z}<extra></extra>";

    // dump the graph parameters to the html string
    html += json_plot.dump();

    // Generate annotations to display "X" symbols on empty plot cells.
    std::string annotationsStr = "[";
    for (int i = 0; i < x_dim; i++) {
      for (int j = 0; j < y_dim; j++) {
        if (resulting_matrix[i][j] == 0) {
          // Generate a single annotation object
          std::string annotation = "{x: " + std::to_string(i) +
                                   ", y: " + std::to_string(j) +
                                   ", text: 'X', showarrow: false}, ";
          annotationsStr += annotation;
        }
      }
    }
    // Remove the trailing comma and space
    if (annotationsStr.length() > 1) {
      annotationsStr = annotationsStr.substr(0, annotationsStr.length() - 2);
    }
    annotationsStr += "]";

    // If x axis cell count is larger than set threshold (25), do not display
    // tick labels to prevent distorted graphs
    nlohmann::json json_xaxis;
    json_xaxis["dtick"] = 1;
    json_xaxis["side"] = "top";
    if (x_dim > 25) {
      json_xaxis["showticklabels"] = false;
    }

    // If y axis cell count is larger than set threshold (25), do not display
    // tick labels to prevent distorted graphs
    nlohmann::json json_yaxis;
    json_yaxis["dtick"] = 1;
    json_yaxis["autorange"] = "reversed";
    if (y_dim > 25) {
      json_yaxis["showticklabels"] = false;
    }

    // Set the layout parameters of the graph.
    // (Might be better to convert it to a json object or another format if
    // further modifications are needed.)
    html += std::string("], {yaxis: ") + json_yaxis.dump() +
            ", xaxis: " + json_xaxis.dump() +
            ", zaxis: {title:\"NNZ(s)\"}, plot_bgcolor:\"rgba(0,0,0,1)\", "
            "paper_bgcolor:\"rgba(0,0,0,0)\", annotations: " +
            annotationsStr;
    html += "}); \nmaximizar();";
    html += "</script> \n";
  }
  html +=
      "  </div>\n"
      "</div>";
}

template <typename IDType, typename NNZType, typename ValueType>
void Visualizer<IDType, NNZType, ValueType>::packHtml() {
  // Finalize the html
  html += "</body> \n </html> \n";
}

template <typename IDType, typename NNZType, typename ValueType>
std::string Visualizer<IDType, NNZType, ValueType>::getHtml() {
  // Returns the generated html
  return html;
}
/*
int main(void) {

    // Example matrices can be found at "https://sparse.tamu.edu/"
    // Below matrix (ecology1.mtx) is from
"https://sparse.tamu.edu/McRae/ecology1"
    // ecology1 matrix' dimensions are 1 mil x 1 mil
    std::string file_name = "ecology1.mtx"; // a matrix in .mtx format in the
same directory sparsebase::io::MTXReader<unsigned int, unsigned int, float>
reader(file_name); sparsebase::format::CSR<unsigned int, unsigned int, float>
*csr = reader.ReadCSR();


    // Alternatively, can generate a CSR ourselves and use it for plotting
    //int csr_row_ptr[5]{0, 2, 3, 3, 4};
    //int csr_col[4]{0, 2, 1, 3};
    //int csr_vals[4]{4, 5, 7, 9};
    //sparsebase::format::CSR<int, int, int> csr(4, 4, csr_row_ptr, csr_col,
csr_vals);


    // Specify the desired orderings to be plotted and add it to the orderings
vector sparsebase::reorder::DegreeReorder<unsigned int,unsigned int,float>
orderer(1); std::vector<sparsebase::reorder::Reorderer<unsigned int>*>
orderings; orderings.push_back(&orderer);

    // Generate a json object for features list, add the feature, value pairs in
the respective sections. nlohmann::json featuresList = {
        // "non_ordering_based_features" contain the ordering invariant
features,
        // which will be displayed at the top of the html page
        {"non_ordering_based_features", {
            {"number_triangles", 0},         // unsigned integer
            {"max_degree", 0},               // unsigned integer
            {"avg_degree", 0.0},             // float
            {"betweenness_centrality", 0.0}, // float
            {"symmetry_ratio", 0.0}          // float
        }},
        // section that contains ordering dependent features names/values and
ordering names for each ordering
        {"features_list", {
            {
                {"order_name", "Natural Ordering"},
                {"features", {
                    {"number_diagonal_entries", 0},            // unsigned
integer
                    {"number_non_diagonal_entries", 0},        // unsigned
integer
                    {"number_dense_rows", 0},                  // unsigned
integer
                    {"bandwidth", 0.0}                         // float
                }}
            },
            {
                {"order_name", "Degree Reorder"},
                {"features", {
                    {"number_diagonal_entries", 0},            // unsigned
integer
                    {"number_non_diagonal_entries", 0},        // unsigned
integer
                    {"number_dense_rows", 0},                  // unsigned
integer
                    {"bandwidth", 0.0}                         // float
                }}
            }
        }}
    };

    // Set bucket size parameter for the plot that will be generated.
    // e.g. For a 1 mil x 1 mil matrix, ceil(1000000/50000) will result in a
20x20 plot matrix
    // Should be set to an appropriate value for the input matrix to avoid too
small/large plot sizes unsigned int bucketSize = 50000;

    // Flag to do plotting by weight or frequency.
    bool doPlotByWeights = true;

    Visualizer<unsigned int,unsigned int,float>vsl (csr, &orderings,
&featuresList, bucketSize, doPlotByWeights);

    std::ofstream file("myfile.html");
    if (file.is_open()) {
        file << vsl.getHtml();
        file.close();
    }
    else {
        std::cout << "Error: Unable to open file!" << std::endl;
    }

    return 0;
}
*/
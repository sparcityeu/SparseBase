#include <string>
#include <set>
#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
#include "sparsebase/object/object.h"
#include "sparsebase/io/metis_graph_reader.h"
TEST(MetisGraphReader, ReadGraph) {
  //Write the metis graph data with vertex & edge weights to a file
  std::ofstream ofs("vertex_edge_weights.graph");
  ofs << metis_graph_1;
  ofs.close();

  //Write the metis graph data with multiple vertex weights to a file
  std::ofstream ofs2("multiple_vertex_weights.graph");
  ofs2 << metis_graph_2;
  ofs2.close();

  //File 1
  //Read it using sparsebase
  sparsebase::io::MetisGraphReader<int, int, int> reader1("vertex_edge_weights.graph");
  sparsebase::object::Graph<int, int, int>* graph1 = reader1.ReadGraph();
  sparsebase::format::Format *con1 = graph1->get_connectivity();
  int n1 = con1->get_dimensions()[0];
  int m1 = con1->get_num_nnz();
  auto row1 =
      con1->AsAbsolute<sparsebase::format::COO<int, int, int>>()->get_row();
  auto col1 = con1->AsAbsolute<sparsebase::format::COO<int, int, int>>()
                 ->get_col();
  auto edgeWeights1 = con1->AsAbsolute<sparsebase::format::COO<int, int, int>>()
                 ->get_vals();
  auto vertexWeights1 = graph1->vertexWeights_;
  //Check the dimensions
  EXPECT_EQ(n1, metis_n_1);
  EXPECT_EQ(m1, metis_m_1);
  EXPECT_EQ(graph1->ncon_, metis_ncon_1);
  EXPECT_NE(row1, nullptr);
  EXPECT_NE(col1, nullptr);
  EXPECT_NE(edgeWeights1, nullptr);
  EXPECT_NE(vertexWeights1, nullptr);

  //Check edge weights
  for (int i = 0; i < m1; ++i)
    EXPECT_EQ(metis_val_1[i], edgeWeights1[i]);

  //Check vertex weights
  for (int vertex = 0; vertex <= n1; ++vertex){
    auto weight = vertexWeights1[vertex]->get_vals()[0];
      EXPECT_EQ(metis_vertex_weights_1[vertex], weight);
    }

  //Check if all edges read correctly
  std::set<std::pair<int, int>> edge_set1;
  for (int i = 0; i < metis_m_1; ++i) {
    edge_set1.emplace(metis_row_1[i], metis_col_1[i]);
  }
  for (int i = 0; i < m1; i++) {
    std::pair<int, int> p(row1[i], col1[i]);
    EXPECT_NE(edge_set1.find(p), edge_set1.end());
  }

  //File 2
  //Read it using sparsebase
  sparsebase::io::MetisGraphReader<int, int, int> reader2("multiple_vertex_weights.graph");
  sparsebase::object::Graph<int, int, int>* graph2 = reader2.ReadGraph();
  sparsebase::format::Format *con2 = graph2->get_connectivity();
  int n2 = con2->get_dimensions()[0];
  int m2 = con2->get_num_nnz();
  auto row2 =
      con2->AsAbsolute<sparsebase::format::COO<int, int, int>>()->get_row();
  auto col2 = con2->AsAbsolute<sparsebase::format::COO<int, int, int>>()
                  ->get_col();
  auto edgeWeights2 = con2->AsAbsolute<sparsebase::format::COO<int, int, int>>()
                          ->get_vals();
  auto vertexWeights2 = graph2->vertexWeights_;
  //Check the dimensions
  EXPECT_EQ(n2, metis_n_2);
  EXPECT_EQ(m2, metis_m_2);
  EXPECT_EQ(graph2->ncon_, metis_ncon_2);
  EXPECT_NE(row2, nullptr);
  EXPECT_NE(col2, nullptr);
  //Graph2 has no edge weights
  EXPECT_EQ(edgeWeights2, nullptr);
  EXPECT_NE(vertexWeights2, nullptr);

  //Check vertex weights
  for (int vertex = 0; vertex <= n2; ++vertex){
    auto weights = vertexWeights2[vertex]->get_vals();
    for (int i = 0; i < metis_ncon_2; ++i){
      EXPECT_EQ(metis_vertex_weights_2[vertex][i], weights[i]);
    }
  }

  //Check if all edges read correctly
  std::set<std::pair<int, int>> edge_set2;
  for (int i = 0; i < metis_m_2; ++i) {
    edge_set2.emplace(metis_row_2[i], metis_col_2[i]);
  }
  for (int i = 0; i < m2; i++) {
    std::pair<int, int> p(row2[i], col2[i]);
    EXPECT_NE(edge_set2.find(p), edge_set2.end());
  }
}

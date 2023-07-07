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
  sparsebase::io::MetisGraphReader<int, int, int> reader1("vertex_edge_weights.graph", false);
  sparsebase::object::Graph<int, int, int>* graph1 = reader1.ReadGraph();
  sparsebase::format::Format *con1 = graph1->get_connectivity();
  int n1 = con1->get_dimensions()[0];
  int m1 = con1->get_num_nnz();
  auto coo1 = con1->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto row1 = coo1->get_row();
  auto col1 = coo1->get_col();
  auto edgeWeights1 = coo1->get_vals();
  auto vertexWeights1 = graph1->vertexWeights_;
  //Check the dimensions
  EXPECT_EQ(n1, metis_n_1 + 1); //1 indexed
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
  for (int vertex = 0; vertex < n1; ++vertex) {
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
  sparsebase::io::MetisGraphReader<int, int, int> reader2("multiple_vertex_weights.graph", false);
  sparsebase::object::Graph<int, int, int>* graph2 = reader2.ReadGraph();
  sparsebase::format::Format *con2 = graph2->get_connectivity();
  int n2 = con2->get_dimensions()[0];
  int m2 = con2->get_num_nnz();
  auto coo2 = con2->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto row2 = coo2->get_row();
  auto col2 = coo2->get_col();
  auto edgeWeights2 = coo2->get_vals();
  auto vertexWeights2 = graph2->vertexWeights_;
  //Check the dimensions
  EXPECT_EQ(n2, metis_n_2 + 1); //1 indexed
  EXPECT_EQ(m2, metis_m_2);
  EXPECT_EQ(graph2->ncon_, metis_ncon_2);
  EXPECT_NE(row2, nullptr);
  EXPECT_NE(col2, nullptr);
  //Graph2 has no edge weights
  EXPECT_EQ(edgeWeights2, nullptr);
  EXPECT_NE(vertexWeights2, nullptr);

  //Check vertex weights
  for (int vertex = 0; vertex < n2; ++vertex) {
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

  //File 1 converted to 0 indexed
  //Read it using sparsebase
  sparsebase::io::MetisGraphReader<int, int, int> reader3("vertex_edge_weights.graph", true);
  sparsebase::object::Graph<int, int, int>* graph3 = reader3.ReadGraph();
  sparsebase::format::Format *con3 = graph3->get_connectivity();
  int n3 = con3->get_dimensions()[0];
  int m3 = con3->get_num_nnz();
  auto coo3 = con3->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto row3 = coo3->get_row();
  auto col3 = coo3->get_col();
  auto edgeWeights3 = coo3->get_vals();
  auto vertexWeights3 = graph3->vertexWeights_;
  //Check the dimensions
  EXPECT_EQ(n3, metis_n_1); //0_indexed
  EXPECT_EQ(m3, metis_m_1);
  EXPECT_EQ(graph3->ncon_, metis_ncon_1);
  EXPECT_NE(row3, nullptr);
  EXPECT_NE(col3, nullptr);
  EXPECT_NE(edgeWeights3, nullptr);
  EXPECT_NE(vertexWeights3, nullptr);

  //Check edge weights
  for (int i = 0; i < m3; ++i)
    EXPECT_EQ(metis_val_1[i], edgeWeights3[i]);

  //Check vertex weights
  for (int vertex = 0; vertex < n3; ++vertex) {
    auto weight = vertexWeights3[vertex]->get_vals()[0];
    EXPECT_EQ(metis_vertex_weights_1[vertex + 1], weight); //0 indexed
  }

  //Check if all edges read correctly
  std::set<std::pair<int, int>> edge_set3;
  for (int i = 0; i < metis_m_1; ++i) {
    edge_set3.emplace(metis_row_1[i] - 1, metis_col_1[i] - 1); //0 indexed
  }
  for (int i = 0; i < m3; i++) {
    std::pair<int, int> p(row3[i], col3[i]);
    EXPECT_NE(edge_set3.find(p), edge_set3.end());
  }

  //File 2 converted to 0 indexed
  //Read it using sparsebase
  sparsebase::io::MetisGraphReader<int, int, int> reader4("multiple_vertex_weights.graph", true);
  sparsebase::object::Graph<int, int, int>* graph4 = reader4.ReadGraph();
  sparsebase::format::Format *con4 = graph4->get_connectivity();
  int n4 = con4->get_dimensions()[0];
  int m4 = con4->get_num_nnz();
  auto coo4 = con4->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto row4 = coo4->get_row();
  auto col4 = coo4->get_col();
  auto edgeWeights4 = coo4->get_vals();
  auto vertexWeights4 = graph4->vertexWeights_;
  //Check the dimensions
  EXPECT_EQ(n4, metis_n_2); //0 indexed
  EXPECT_EQ(m4, metis_m_2);
  EXPECT_EQ(graph4->ncon_, metis_ncon_2);
  EXPECT_NE(row4, nullptr);
  EXPECT_NE(col4, nullptr);
  //Graph2 has no edge weights
  EXPECT_EQ(edgeWeights4, nullptr);
  EXPECT_NE(vertexWeights4, nullptr);

  //Check vertex weights
  for (int vertex = 0; vertex < n4; ++vertex) {
    auto weights = vertexWeights4[vertex]->get_vals();
    for (int i = 0; i < metis_ncon_2; ++i){
      EXPECT_EQ(metis_vertex_weights_2[vertex + 1][i], weights[i]); //0 indexed
    }
  }

  //Check if all edges read correctly
  std::set<std::pair<int, int>> edge_set4;
  for (int i = 0; i < metis_m_2; ++i) {
    edge_set4.emplace(metis_row_2[i] - 1, metis_col_2[i] - 1); //0 indexed
  }
  for (int i = 0; i < m4; i++) {
    std::pair<int, int> p(row4[i], col4[i]);
    EXPECT_NE(edge_set4.find(p), edge_set4.end());
  }

  //File 1 with ValueType void/no weights
  sparsebase::io::MetisGraphReader<int, int, void> reader5("vertex_edge_weights.graph", false);
  sparsebase::object::Graph<int, int, void>* graph5 = reader5.ReadGraph();
  sparsebase::format::Format *con5 = graph5->get_connectivity();
  int n5 = con5->get_dimensions()[0];
  int m5 = con5->get_num_nnz();
  auto coo5 = con5->AsAbsolute<sparsebase::format::COO<int, int, void>>();
  auto row5 = coo5->get_row();
  auto col5 = coo5->get_col();
  //Check the dimensions
  EXPECT_EQ(n5, metis_n_1 + 1); //1 indexed
  EXPECT_EQ(m5, metis_m_1);
  EXPECT_EQ(graph5->ncon_, 0);
  EXPECT_NE(row5, nullptr);
  EXPECT_NE(col5, nullptr);
  EXPECT_EQ(coo5->get_vals(), nullptr);
  EXPECT_EQ(graph5->vertexWeights_, nullptr);

  //Check if all edges read correctly
  std::set<std::pair<int, int>> edge_set5;
  for (int i = 0; i < metis_m_1; ++i) {
    edge_set5.emplace(metis_row_1[i], metis_col_1[i]);
  }
  for (int i = 0; i < m5; i++) {
    std::pair<int, int> p(row5[i], col5[i]);
    EXPECT_NE(edge_set5.find(p), edge_set5.end());
  }
}

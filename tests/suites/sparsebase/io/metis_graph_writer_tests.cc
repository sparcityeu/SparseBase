#include <string>
#include <set>
#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
#include "sparsebase/object/object.h"
#include "sparsebase/io/metis_graph_reader.h"
#include "sparsebase/io/metis_graph_writer.h"
TEST(MetisGraphWriter, WriteGraph) {
  //Write the metis graph data with vertex & edge weights to a file
  std::ofstream ofs("vertex_edge_weights.graph");
  ofs << metis_graph_1;
  ofs.close();

  //Write the metis graph data with multiple vertex weights to a file
  std::ofstream ofs2("multiple_vertex_weights.graph");
  ofs2 << metis_graph_2;
  ofs2.close();

  //File 1
  //Read the original graph 1 from data
  sparsebase::io::MetisGraphReader<int, int, int> org_reader1("vertex_edge_weights.graph", false);
  sparsebase::object::Graph<int, int, int>* org_graph1 = org_reader1.ReadGraph();
  sparsebase::format::Format *org_con1 = org_graph1->get_connectivity();
  int org_n1 = org_con1->get_dimensions()[0];
  int org_m1 = org_con1->get_num_nnz();
  int org_ncon1 = org_graph1->ncon_;
  auto org_coo1 = org_con1->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto org_row1 = org_coo1->get_row();
  auto org_col1 = org_coo1->get_col();
  auto org_val1 = org_coo1->get_vals();
  auto org_vertexWeights1 = org_graph1->vertexWeights_;

  //Write the original graph 1
  sparsebase::io::MetisGraphWriter<int, int, int> writer1("org1.graph",
                                                          true,true,false);
  writer1.WriteGraph(org_graph1);
  //Read the output of writer
  sparsebase::io::MetisGraphReader<int, int, int> written_reader1("org1.graph", false);
  sparsebase::object::Graph<int, int, int>* written_graph1 = written_reader1.ReadGraph();
  sparsebase::format::Format *written_con1 = written_graph1->get_connectivity();
  int written_n1 = written_con1->get_dimensions()[0];
  int written_m1 = written_con1->get_num_nnz();
  int written_ncon1 = written_graph1->ncon_;
  auto written_coo1 = written_con1->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto written_row1 = written_coo1->get_row();
  auto written_col1 = written_coo1->get_col();
  auto written_val1 = written_coo1->get_vals();
  auto written_vertexWeights1 = written_graph1->vertexWeights_;

  //Compare the original graph 1 with the written version
  //Check the dimensions
  EXPECT_EQ(org_n1, written_n1);
  EXPECT_EQ(org_m1, written_m1);
  EXPECT_EQ(org_ncon1, written_ncon1);
  //Check edges and weights
  for (int i = 0; i < org_coo1->get_num_nnz(); ++i) {
    EXPECT_EQ(org_row1[i], written_row1[i]);
    EXPECT_EQ(org_col1[i], written_col1[i]);
    EXPECT_EQ(org_val1[i], written_val1[i]);
  }
  //Check vertex weights
  for (int vertex = 0; vertex < org_n1; ++vertex) {
    auto org_weights = org_vertexWeights1[vertex]->get_vals();
    auto written_weights = written_vertexWeights1[vertex]->get_vals();
    for (int j = 0 ;j < org_ncon1; ++j) {
      EXPECT_EQ(org_weights[j], written_weights[j]);
    }
  }

  //File 2
  //Read the original graph 2 from data
  sparsebase::io::MetisGraphReader<int, int, int> org_reader2("multiple_vertex_weights.graph", false);
  sparsebase::object::Graph<int, int, int>* org_graph2 = org_reader2.ReadGraph();
  sparsebase::format::Format *org_con2 = org_graph2->get_connectivity();
  int org_n2 = org_con2->get_dimensions()[0];
  int org_m2 = org_con2->get_num_nnz();
  int org_ncon2 = org_graph2->ncon_;
  auto org_coo2 = org_con2->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto org_row2= org_coo2->get_row();
  auto org_col2 = org_coo2->get_col();
  EXPECT_EQ(org_coo2->get_vals(), nullptr);
  auto org_vertexWeights2 = org_graph2->vertexWeights_;

  //Write the original graph 2
  sparsebase::io::MetisGraphWriter<int, int, int> writer2("org2.graph",
                                                          false,true,false);
  writer2.WriteGraph(org_graph2);
  //Read the output of writer
  sparsebase::io::MetisGraphReader<int, int, int> written_reader2("org2.graph", false);
  sparsebase::object::Graph<int, int, int>* written_graph2 = written_reader2.ReadGraph();
  sparsebase::format::Format *written_con2 = written_graph2->get_connectivity();
  int written_n2 = written_con2->get_dimensions()[0];
  int written_m2 = written_con2->get_num_nnz();
  int written_ncon2 = written_graph2->ncon_;
  auto written_coo2 = written_con2->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto written_row2 = written_coo2->get_row();
  auto written_col2 = written_coo2->get_col();
  auto written_vertexWeights2 = written_graph2->vertexWeights_;

  //Compare the original graph 2 with the written version
  //Check the dimensions
  EXPECT_EQ(org_n2, written_n2);
  EXPECT_EQ(org_m2, written_m2);
  EXPECT_EQ(org_ncon2, written_ncon2);
  //Check edges
  for (int i = 0; i < org_coo2->get_num_nnz(); ++i) {
      EXPECT_EQ(org_row2[i], written_row2[i]);
      EXPECT_EQ(org_col2[i], written_col2[i]);
  }
  //Check vertex weights
  for (int vertex = 0; vertex < org_n2; ++vertex) {
      auto org_weights = org_vertexWeights2[vertex]->get_vals();
      auto written_weights = written_vertexWeights2[vertex]->get_vals();
      for (int j = 0 ;j < org_ncon2; ++j) {
        EXPECT_EQ(org_weights[j], written_weights[j]);
      }
  }

  //File 1, write from 0 based indices
  //Read the original graph 3 from data
  sparsebase::io::MetisGraphReader<int, int, int> org_reader3("vertex_edge_weights.graph", true);
  sparsebase::object::Graph<int, int, int>* org_graph3 = org_reader3.ReadGraph();
  sparsebase::format::Format *org_con3 = org_graph3->get_connectivity();
  int org_n3 = org_con3->get_dimensions()[0];
  int org_m3 = org_con3->get_num_nnz();
  int org_ncon3 = org_graph3->ncon_;
  auto org_coo3 = org_con3->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto org_row3 = org_coo3->get_row();
  auto org_col3 = org_coo3->get_col();
  auto org_val3 = org_coo3->get_vals();
  auto org_vertexWeights3 = org_graph3->vertexWeights_;

  //Write the original graph 3
  sparsebase::io::MetisGraphWriter<int, int, int> writer3("org3.graph",
                                                          true,true,true);
  writer3.WriteGraph(org_graph3);
  //Read the output of writer
  sparsebase::io::MetisGraphReader<int, int, int> written_reader3("org3.graph", true);
  sparsebase::object::Graph<int, int, int>* written_graph3 = written_reader3.ReadGraph();
  sparsebase::format::Format *written_con3 = written_graph3->get_connectivity();
  int written_n3 = written_con3->get_dimensions()[0];
  int written_m3 = written_con3->get_num_nnz();
  int written_ncon3 = written_graph3->ncon_;
  auto written_coo3 = written_con3->AsAbsolute<sparsebase::format::COO<int, int, int>>();
  auto written_row3 = written_coo3->get_row();
  auto written_col3 = written_coo3->get_col();
  auto written_val3 = written_coo3->get_vals();
  auto written_vertexWeights3 = written_graph3->vertexWeights_;

  //Compare the original graph 3 with the written version
  //Check the dimensions
  EXPECT_EQ(org_n3, written_n3);
  EXPECT_EQ(org_m3, written_m3);
  EXPECT_EQ(org_ncon3, written_ncon3);
  //Check edges and weights
  for (int i = 0; i < org_coo3->get_num_nnz(); ++i) {
      EXPECT_EQ(org_row3[i], written_row3[i]);
      EXPECT_EQ(org_col3[i], written_col3[i]);
      EXPECT_EQ(org_val3[i], written_val3[i]);
  }
  //Check vertex weights
  for (int vertex = 0; vertex < org_n3; ++vertex) {
      auto org_weights = org_vertexWeights3[vertex]->get_vals();
      auto written_weights = written_vertexWeights3[vertex]->get_vals();
      for (int j = 0 ;j < org_ncon3; ++j) {
        EXPECT_EQ(org_weights[j], written_weights[j]);
      }
  }

  //File 1 with ValueType void/no weights
  //Read the original graph 4 from data
  sparsebase::io::MetisGraphReader<int, int, void> org_reader4("vertex_edge_weights.graph", false);
  sparsebase::object::Graph<int, int, void>* org_graph4 = org_reader4.ReadGraph();
  sparsebase::format::Format *org_con4 = org_graph4->get_connectivity();
  int org_n4 = org_con4->get_dimensions()[0];
  int org_m4 = org_con4->get_num_nnz();
  EXPECT_EQ(org_graph4->ncon_, 0);
  auto org_coo4 = org_con4->AsAbsolute<sparsebase::format::COO<int, int, void>>();
  auto org_row4 = org_coo4->get_row();
  auto org_col4 = org_coo4->get_col();
  EXPECT_EQ(org_coo4->get_vals(), nullptr);
  EXPECT_EQ(org_graph4->vertexWeights_, nullptr);

  //Write the original graph 4
  sparsebase::io::MetisGraphWriter<int, int, void> writer4("org4.graph",
                                                          false,false,false);
  writer4.WriteGraph(org_graph4);
  //Read the output of writer
  sparsebase::io::MetisGraphReader<int, int, void> written_reader4("org4.graph", false);
  sparsebase::object::Graph<int, int, void>* written_graph4 = written_reader4.ReadGraph();
  sparsebase::format::Format *written_con4 = written_graph4->get_connectivity();
  int written_n4 = written_con4->get_dimensions()[0];
  int written_m4 = written_con4->get_num_nnz();
  EXPECT_EQ(written_graph4->ncon_,0);
  auto written_coo4 = written_con4->AsAbsolute<sparsebase::format::COO<int, int, void>>();
  auto written_row4 = written_coo4->get_row();
  auto written_col4 = written_coo4->get_col();
  EXPECT_EQ(written_coo4->get_vals(), nullptr);
  EXPECT_EQ(written_graph4->vertexWeights_, nullptr);

  //Compare the original graph 4 with the written version
  //Check the dimensions
  EXPECT_EQ(org_n4, written_n4);
  EXPECT_EQ(org_m4, written_m4);

  //Check edges
  for (int i = 0; i < org_coo4->get_num_nnz(); ++i) {
      EXPECT_EQ(org_row4[i], written_row4[i]);
      EXPECT_EQ(org_col4[i], written_col4[i]);
  }

}

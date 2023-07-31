#include <string>
#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
#include "sparsebase/object/object.h"
#include "sparsebase/io/patoh_reader.h"

TEST(PatohReader,ReadHyperGraph1){
  //Write the hypergraph data with no edge and vertices weight to a file
    std::ofstream ofs("HyperGraph_no_Edge_and_Vertices_Weight.hypeg");
    ofs << hypergraph_1;
    ofs.close();

    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph1("HyperGraph_no_Edge_and_Vertices_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* hypergraph1 = ReadHyperGraph1.ReadHyperGraph();
    sparsebase::format::Format *con = hypergraph1->get_connectivity();
    int n_ = hypergraph1->n_;
    int m_ = hypergraph1->m_;
    int constraint_num_ = hypergraph1->constraint_num_;
    int base_type_ = hypergraph1->base_type_;
    int vertex_size = con->get_dimensions()[1];
    auto xpins = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto pins = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto xpin_val_arr = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto xNetsCSR = hypergraph1->xNetCSR_;
    auto xnets = xNetsCSR->get_row_ptr();
    auto cells = xNetsCSR->get_col();
    auto xnet_val_arr = xNetsCSR->get_vals();
    auto netWeights = hypergraph1->netWeights_;
    auto cellWeights = hypergraph1->cellWeights_;

    //Check the dimensions
    EXPECT_EQ(n_,hypergraph1_n);
    EXPECT_EQ(vertex_size, hypergraph1_vertex_size);
    EXPECT_EQ(m_,hypergraph1_m);
    EXPECT_EQ(constraint_num_,hypergraph1_constraint_num);
    EXPECT_EQ(base_type_,hypergraph1_base_type);
    EXPECT_NE(xpins, nullptr);
    EXPECT_NE(pins, nullptr);
    EXPECT_NE(xnets, nullptr);
    EXPECT_NE(cells, nullptr);
    EXPECT_NE(netWeights, nullptr);
    EXPECT_NE(cellWeights, nullptr);
    EXPECT_EQ(xpin_val_arr, nullptr);
    EXPECT_EQ(xnet_val_arr, nullptr);

    //Check xpins
    for (int i = 0; i < n_+1; ++i)
        EXPECT_EQ(hypergraph1_xpins[i], xpins[i]);

    //Check pins
    for (int i = 0; i < m_; ++i)
        EXPECT_EQ(hypergraph1_pins[i], pins[i]);

    //Check edge weights
    auto netWeightsVal = netWeights->get_vals(); 
    for (int i = 0; i < n_; ++i)
        EXPECT_EQ(hypergraph1_netWeights[i], netWeightsVal[i]);

    //Check cell weights
    auto cellWeightsVal = cellWeights->get_vals(); 
    for (int i = 0; i < vertex_size; ++i)
        EXPECT_EQ(hypergraph1_cellWeights[i], cellWeightsVal[i]);

    // Check xnets
    for (int i = 0; i<vertex_size+1;++i)
         EXPECT_EQ(hypergraph1_xnets[i], xnets[i]);
    
    // Check cells
    for (int i = 0; i<m_;++i)
         EXPECT_EQ(hypergraph1_cells[i], cells[i]);


     //Write the hypergraph data with no vertices weight to a file
    std::ofstream ofs2("HyperGraph_no_Vertices_Weight.hypeg");
    ofs2 << hypergraph_2;
    ofs2.close();

    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph2("HyperGraph_no_Vertices_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* hypergraph2 = ReadHyperGraph2.ReadHyperGraph();
    sparsebase::format::Format *con2 = hypergraph2->get_connectivity();
    int n2_ = hypergraph2->n_;
    int m2_ = hypergraph2->m_;
    int vertex_size2 = con2->get_dimensions()[1];
    auto xpins2 = con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto pins2 = con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto xNetsCSR2 = hypergraph2->xNetCSR_;
    auto xnets2 = xNetsCSR2->get_row_ptr();
    auto cells2 = xNetsCSR2->get_col();
    auto xnet_val_arr2 = xNetsCSR2->get_vals();
    auto xpin_val_arr2 = con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto netWeights2 = hypergraph2->netWeights_;
    auto cellWeights2 = hypergraph2->cellWeights_;

    //Check the dimensions
    EXPECT_EQ(n2_,hypergraph2_n);
    EXPECT_EQ(vertex_size2, hypergraph2_vertex_size);
    EXPECT_EQ(m2_,hypergraph2_m);
    EXPECT_NE(xpins2, nullptr);
    EXPECT_NE(pins2, nullptr);
    EXPECT_NE(xnets2, nullptr);
    EXPECT_NE(cells2, nullptr);
    EXPECT_NE(netWeights2, nullptr);
    EXPECT_NE(cellWeights2, nullptr);
    EXPECT_EQ(xpin_val_arr2, nullptr);
    EXPECT_EQ(xnet_val_arr2, nullptr);

    //Check xpins
    for (int i = 0; i < n2_+1; ++i)
        EXPECT_EQ(hypergraph2_xpins[i], xpins2[i]);

    //Check pins
    for (int i = 0; i < m2_; ++i)
        EXPECT_EQ(hypergraph2_pins[i], pins2[i]);

    //Check edge weights
    auto netWeightsVal2 = netWeights2->get_vals(); 
    for (int i = 0; i < n2_; ++i)
        EXPECT_EQ(hypergraph2_netWeights[i], netWeightsVal2[i]);

    //Check cell weights
    auto cellWeightsVal2 = cellWeights2->get_vals(); 
    for (int i = 0; i < vertex_size2; ++i)
        EXPECT_EQ(hypergraph2_cellWeights[i], cellWeightsVal2[i]);

    // Check xnets
    for (int i = 0; i<(vertex_size2)+1;++i)
         EXPECT_EQ(hypergraph2_xnets[i], xnets2[i]);
    
    // Check cells
    for (int i = 0; i<m2_;++i)
         EXPECT_EQ(hypergraph2_cells[i], cells2[i]);

    

     //Write the hypergraph data with no nets weight to a file
    std::ofstream ofs3("HyperGraph_no_Nets_Weight.hypeg");
    ofs3 << hypergraph_3;
    ofs3.close();

    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph3("HyperGraph_no_Nets_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* hypergraph3 = ReadHyperGraph3.ReadHyperGraph();
    sparsebase::format::Format *con3 = hypergraph3->get_connectivity();
    int n3_ = hypergraph3->n_;
    int m3_ = hypergraph3->m_;
    int vertex_size3 = con3->get_dimensions()[1];
    auto xpins3 = con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto pins3 = con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto xNetsCSR3 = hypergraph3->xNetCSR_;
    auto xnets3 = xNetsCSR3->get_row_ptr();
    auto cells3 = xNetsCSR3->get_col();
    auto xnet_val_arr3 = xNetsCSR3->get_vals();
    auto xpin_val_arr3 = con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto netWeights3 = hypergraph3->netWeights_;
    auto cellWeights3 = hypergraph3->cellWeights_;

    //Check the dimensions
    EXPECT_EQ(n3_,hypergraph3_n);
    EXPECT_EQ(vertex_size3, hypergraph3_vertex_size);
    EXPECT_EQ(m3_,hypergraph3_m);
    EXPECT_NE(xpins3, nullptr);
    EXPECT_NE(pins3, nullptr);
    EXPECT_NE(xnets3, nullptr);
    EXPECT_NE(cells3, nullptr);
    EXPECT_NE(netWeights3, nullptr);
    EXPECT_NE(cellWeights3, nullptr);
    EXPECT_EQ(xpin_val_arr3, nullptr);
    EXPECT_EQ(xnet_val_arr3, nullptr);

    //Check xpins
    for (int i = 0; i < n3_+1; ++i)
        EXPECT_EQ(hypergraph3_xpins[i], xpins3[i]);

    //Check pins
    for (int i = 0; i < m3_; ++i)
        EXPECT_EQ(hypergraph3_pins[i], pins3[i]);

    //Check edge weights
    auto netWeightsVal3 = netWeights3->get_vals(); 
    for (int i = 0; i < n3_; ++i)
        EXPECT_EQ(hypergraph3_netWeights[i], netWeightsVal3[i]);

    //Check cell weights
    auto cellWeightsVal3 = cellWeights3->get_vals(); 
    for (int i = 0; i < vertex_size3; ++i)
        EXPECT_EQ(hypergraph3_cellWeights[i], cellWeightsVal3[i]);

    // Check xnets
    for (int i = 0; i<(vertex_size3)+1;++i)
         EXPECT_EQ(hypergraph3_xnets[i], xnets3[i]);
    
    // Check cells
    for (int i = 0; i<m3_;++i)
         EXPECT_EQ(hypergraph3_cells[i], cells3[i]);

    

    //Write the hypergraph data with both vertices and nets weight to a file
    std::ofstream ofs4("HyperGraph_With_Cell_and_Nets_Weight.hypeg");
    ofs4 << hypergraph_4;
    ofs4.close();

    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph4("HyperGraph_With_Cell_and_Nets_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* hypergraph4 = ReadHyperGraph4.ReadHyperGraph();
    sparsebase::format::Format *con4 = hypergraph4->get_connectivity();
    int n4_ = hypergraph4->n_;
    int m4_ = hypergraph4->m_;
    int vertex_size4 = con4->get_dimensions()[1];
    auto xpins4 = con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto pins4 = con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto xNetsCSR4 = hypergraph4->xNetCSR_;
    auto xnets4 = xNetsCSR4->get_row_ptr();
    auto cells4 = xNetsCSR4->get_col();
    auto xnet_val_arr4 = xNetsCSR4->get_vals();
    auto xpin_val_arr4 = con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto netWeights4 = hypergraph4->netWeights_;
    auto cellWeights4 = hypergraph4->cellWeights_;

    //Check the dimensions
    EXPECT_EQ(n4_,hypergraph4_n);
    EXPECT_EQ(vertex_size4, hypergraph4_vertex_size);
    EXPECT_EQ(m4_,hypergraph4_m);
    EXPECT_NE(xpins4, nullptr);
    EXPECT_NE(pins4, nullptr);
    EXPECT_NE(xnets4, nullptr);
    EXPECT_NE(cells4, nullptr);
    EXPECT_NE(netWeights4, nullptr);
    EXPECT_NE(cellWeights4, nullptr);
    EXPECT_EQ(xpin_val_arr4, nullptr);
    EXPECT_EQ(xnet_val_arr4, nullptr);

    //Check xpins
    for (int i = 0; i < n4_+1; ++i)
        EXPECT_EQ(hypergraph4_xpins[i], xpins4[i]);

    //Check pins
    for (int i = 0; i < m4_; ++i)
        EXPECT_EQ(hypergraph4_pins[i], pins4[i]);

    //Check edge weights
    auto netWeightsVal4 = netWeights4->get_vals(); 
    for (int i = 0; i < n4_; ++i)
        EXPECT_EQ(hypergraph4_netWeights[i], netWeightsVal4[i]);

    //Check cell weights
    auto cellWeightsVal4 = cellWeights4->get_vals(); 
    for (int i = 0; i < vertex_size4; ++i)
        EXPECT_EQ(hypergraph4_cellWeights[i], cellWeightsVal4[i]);

    // Check xnets
    for (int i = 0; i<(vertex_size4)+1;++i)
         EXPECT_EQ(hypergraph4_xnets[i], xnets4[i]);
    
    // Check cells
    for (int i = 0; i<m4_;++i)
         EXPECT_EQ(hypergraph4_cells[i], cells4[i]);
}

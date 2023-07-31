#include <string>
#include "gtest/gtest.h"
#include "reader_data.inc"
#include "sparsebase/sparsebase.h"
#include "sparsebase/object/object.h"
#include "sparsebase/io/patoh_reader.h"
#include "sparsebase/io/patoh_writer.h"

TEST(PatohWriter,WriteHyperGraph){
    //Write the hypergraph data with no edge and vertices weight to a file
    std::ofstream ofs("HyperGraph_no_Edge_and_Vertices_Weight.hypeg");
    ofs << hypergraph_1;
    ofs.close();

    //Read the original hypergraph1 from data
    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph1("HyperGraph_no_Edge_and_Vertices_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* org_hypergraph1 = ReadHyperGraph1.ReadHyperGraph();
    sparsebase::format::Format *org_con = org_hypergraph1->get_connectivity();
    int org_n1 = org_hypergraph1->n_;
    int org_m1 = org_hypergraph1->m_;
    int org_constraint_num1 = org_hypergraph1->constraint_num_;
    int org_base_type1 = org_hypergraph1->base_type_;
    int org_vertex_size1 = org_con->get_dimensions()[1];
    auto org_xpins1 = org_con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto org_pins1 = org_con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto org_xpin_val_arr1 = org_con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto org_xNetsCSR1 = org_hypergraph1->xNetCSR_;
    auto org_xnets1 = org_xNetsCSR1->get_row_ptr();
    auto org_cells1 = org_xNetsCSR1->get_col();
    auto org_xnet_val_arr1 = org_xNetsCSR1->get_vals();
    auto org_netWeights1 = org_hypergraph1->netWeights_;
    auto org_cellWeights1 = org_hypergraph1->cellWeights_;

    //Write the original graph1
    sparsebase::io::PatohWriter<int,int,int> WriteHyperGraph1("org_hypergraph1.hypeg",true);
    WriteHyperGraph1.WriteHyperGraph(org_hypergraph1);

    //Read the output of writer
    sparsebase::io::PatohReader<int,int,int> Read_Written_HyperGraph1("org_hypergraph1.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* written_hypergraph1 = Read_Written_HyperGraph1.ReadHyperGraph();
    sparsebase::format::Format *written_con = written_hypergraph1->get_connectivity();

    int written_n1 = written_hypergraph1->n_;
    int written_m1 = written_hypergraph1->m_;
    int written_constraint_num1 = written_hypergraph1->constraint_num_;
    int written_base_type1 = written_hypergraph1->base_type_;
    int written_vertex_size1 = written_con->get_dimensions()[1];
    auto written_xpins1 = written_con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto written_pins1 = written_con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto written_xpin_val_arr1 = written_con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto written_xNetsCSR1 = written_hypergraph1->xNetCSR_;
    auto written_xnets1 = written_xNetsCSR1->get_row_ptr();
    auto written_cells1 = written_xNetsCSR1->get_col();
    auto written_xnet_val_arr1 = written_xNetsCSR1->get_vals();
    auto written_netWeights1 = written_hypergraph1->netWeights_;
    auto written_cellWeights1 = written_hypergraph1->cellWeights_;

    //Compare the original hypergraph 1 with the written version
    //Check the dimensions
    EXPECT_EQ(org_n1,written_n1);
    EXPECT_EQ(org_vertex_size1, written_vertex_size1);
    EXPECT_EQ(org_m1,written_m1);
    EXPECT_EQ(org_constraint_num1,written_constraint_num1);
    EXPECT_EQ(org_base_type1,written_base_type1);

    EXPECT_EQ(written_xpin_val_arr1, nullptr);
    EXPECT_EQ(written_xnet_val_arr1, nullptr);

     //Check xpins
    for (int i = 0; i < org_n1+1; ++i)
        EXPECT_EQ(org_xpins1[i], written_xpins1[i]);

    //Check pins
    for (int i = 0; i < org_m1; ++i)
        EXPECT_EQ(org_pins1[i], written_pins1[i]);

    //Check edge weights
    auto org_netWeightsVal1 = org_netWeights1->get_vals();
    auto written_netWeightsVal1 = written_netWeights1->get_vals();
    for (int i = 0; i < org_n1; ++i)
        EXPECT_EQ(org_netWeightsVal1[i], written_netWeightsVal1[i]);

    //Check cell weights
    auto org_cellWeightsVal1 = org_cellWeights1->get_vals();
    auto written_cellWeightsVal1 = written_cellWeights1->get_vals(); 

    for (int i = 0; i < org_vertex_size1; ++i)
        EXPECT_EQ(org_cellWeightsVal1[i], written_cellWeightsVal1[i]);

    // Check xnets
    for (int i = 0; i<org_vertex_size1+1;++i)
         EXPECT_EQ(org_xnets1[i], written_xnets1[i]);
    
    // Check cells
    for (int i = 0; i<org_m1;++i)
         EXPECT_EQ(org_cells1[i], written_cells1[i]);

    // TEST2

    //Write the hypergraph data with no vertices weight to a file
    std::ofstream ofs2("HyperGraph_no_Vertices_Weight.hypeg");
    ofs2 << hypergraph_2;
    ofs2.close();

    //Read the original hypergraph2 from data
    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph2("HyperGraph_no_Vertices_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* org_hypergraph2 = ReadHyperGraph2.ReadHyperGraph();
    sparsebase::format::Format *org_con2 = org_hypergraph2->get_connectivity();
    int org_n2 = org_hypergraph2->n_;
    int org_m2 = org_hypergraph2->m_;
    int org_constraint_num2 = org_hypergraph2->constraint_num_;
    int org_base_type2 = org_hypergraph2->base_type_;
    int org_vertex_size2 = org_con2->get_dimensions()[1];
    auto org_xpins2 = org_con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto org_pins2 = org_con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto org_xpin_val_arr2 = org_con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto org_xNetsCSR2 = org_hypergraph2->xNetCSR_;
    auto org_xnets2 = org_xNetsCSR2->get_row_ptr();
    auto org_cells2 = org_xNetsCSR2->get_col();
    auto org_xnet_val_arr2 = org_xNetsCSR2->get_vals();
    auto org_netWeights2 = org_hypergraph2->netWeights_;
    auto org_cellWeights2 = org_hypergraph2->cellWeights_;

     //Write the original graph2
    sparsebase::io::PatohWriter<int,int,int> WriteHyperGraph2("org_hypergraph2.hypeg",false,true,false);
    WriteHyperGraph2.WriteHyperGraph(org_hypergraph2);

    //Read the output of writer
    sparsebase::io::PatohReader<int,int,int> Read_Written_HyperGraph2("org_hypergraph2.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* written_hypergraph2 = Read_Written_HyperGraph2.ReadHyperGraph();
    sparsebase::format::Format *written_con2 = written_hypergraph2->get_connectivity();

    int written_n2 = written_hypergraph2->n_;
    int written_m2 = written_hypergraph2->m_;
    int written_constraint_num2 = written_hypergraph2->constraint_num_;
    int written_base_type2 = written_hypergraph2->base_type_;
    int written_vertex_size2 = written_con2->get_dimensions()[1];
    auto written_xpins2 = written_con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto written_pins2 = written_con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto written_xpin_val_arr2 = written_con2->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto written_xNetsCSR2 = written_hypergraph2->xNetCSR_;
    auto written_xnets2 = written_xNetsCSR2->get_row_ptr();
    auto written_cells2 = written_xNetsCSR2->get_col();
    auto written_xnet_val_arr2 = written_xNetsCSR2->get_vals();
    auto written_netWeights2 = written_hypergraph2->netWeights_;
    auto written_cellWeights2 = written_hypergraph2->cellWeights_;

    //Compare the original hypergraph 2 with the written version
    //Check the dimensions
    EXPECT_EQ(org_n2,written_n2);
    EXPECT_EQ(org_vertex_size2, written_vertex_size2);
    EXPECT_EQ(org_m2,written_m2);
    EXPECT_EQ(org_constraint_num2,written_constraint_num2);
    EXPECT_EQ(org_base_type2,written_base_type2);

    EXPECT_EQ(written_xpin_val_arr2, nullptr);
    EXPECT_EQ(written_xnet_val_arr2, nullptr);

     //Check xpins
    for (int i = 0; i < org_n2+1; ++i)
        EXPECT_EQ(org_xpins2[i], written_xpins2[i]);

    //Check pins
    for (int i = 0; i < org_m2; ++i)
        EXPECT_EQ(org_pins2[i], written_pins2[i]);

    //Check edge weights
    auto org_netWeightsVal2 = org_netWeights2->get_vals();
    auto written_netWeightsVal2 = written_netWeights2->get_vals();
    for (int i = 0; i < org_n2; ++i)
        EXPECT_EQ(org_netWeightsVal2[i], written_netWeightsVal2[i]);

    //Check cell weights
    auto org_cellWeightsVal2 = org_cellWeights2->get_vals();
    auto written_cellWeightsVal2 = written_cellWeights2->get_vals(); 

    for (int i = 0; i < org_vertex_size2; ++i)
        EXPECT_EQ(org_cellWeightsVal2[i], written_cellWeightsVal2[i]);

    // Check xnets
    for (int i = 0; i<org_vertex_size2+1;++i)
         EXPECT_EQ(org_xnets2[i], written_xnets2[i]);
    
    // Check cells
    for (int i = 0; i<org_m2;++i)
         EXPECT_EQ(org_cells2[i], written_cells2[i]);

    // TEST3

    //Write the hypergraph data with no nets weight to a file
    std::ofstream ofs3("HyperGraph_no_Nets_Weight.hypeg");
    ofs3 << hypergraph_3;
    ofs3.close();

    //Read the original hypergraph3 from data
    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph3("HyperGraph_no_Nets_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* org_hypergraph3 = ReadHyperGraph3.ReadHyperGraph();
    sparsebase::format::Format *org_con3 = org_hypergraph3->get_connectivity();
    int org_n3 = org_hypergraph3->n_;
    int org_m3 = org_hypergraph3->m_;
    int org_constraint_num3 = org_hypergraph3->constraint_num_;
    int org_base_type3 = org_hypergraph3->base_type_;
    int org_vertex_size3 = org_con3->get_dimensions()[1];
    auto org_xpins3 = org_con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto org_pins3 = org_con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto org_xpin_val_arr3 = org_con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto org_xNetsCSR3 = org_hypergraph3->xNetCSR_;
    auto org_xnets3 = org_xNetsCSR3->get_row_ptr();
    auto org_cells3 = org_xNetsCSR3->get_col();
    auto org_xnet_val_arr3 = org_xNetsCSR3->get_vals();
    auto org_netWeights3 = org_hypergraph3->netWeights_;
    auto org_cellWeights3 = org_hypergraph3->cellWeights_;

    //Write the original graph3
    sparsebase::io::PatohWriter<int,int,int> WriteHyperGraph3("org_hypergraph3.hypeg",false,false,true);
    WriteHyperGraph3.WriteHyperGraph(org_hypergraph3);

    //Read the output of writer
    sparsebase::io::PatohReader<int,int,int> Read_Written_HyperGraph3("org_hypergraph3.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* written_hypergraph3 = Read_Written_HyperGraph3.ReadHyperGraph();
    sparsebase::format::Format *written_con3 = written_hypergraph3->get_connectivity();

    int written_n3 = written_hypergraph3->n_;
    int written_m3 = written_hypergraph3->m_;
    int written_constraint_num3 = written_hypergraph3->constraint_num_;
    int written_base_type3 = written_hypergraph3->base_type_;
    int written_vertex_size3 = written_con3->get_dimensions()[1];
    auto written_xpins3 = written_con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto written_pins3 = written_con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto written_xpin_val_arr3 = written_con3->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto written_xNetsCSR3 = written_hypergraph3->xNetCSR_;
    auto written_xnets3 = written_xNetsCSR3->get_row_ptr();
    auto written_cells3 = written_xNetsCSR3->get_col();
    auto written_xnet_val_arr3 = written_xNetsCSR3->get_vals();
    auto written_netWeights3 = written_hypergraph3->netWeights_;
    auto written_cellWeights3 = written_hypergraph3->cellWeights_;

    //Compare the original hypergraph 3 with the written version
    //Check the dimensions
    EXPECT_EQ(org_n3,written_n3);
    EXPECT_EQ(org_vertex_size3, written_vertex_size3);
    EXPECT_EQ(org_m3,written_m3);
    EXPECT_EQ(org_constraint_num3,written_constraint_num3);
    EXPECT_EQ(org_base_type3,written_base_type3);

    EXPECT_EQ(written_xpin_val_arr3, nullptr);
    EXPECT_EQ(written_xnet_val_arr3, nullptr);

     //Check xpins
    for (int i = 0; i < org_n3+1; ++i)
        EXPECT_EQ(org_xpins3[i], written_xpins3[i]);

    //Check pins
    for (int i = 0; i < org_m3; ++i)
        EXPECT_EQ(org_pins3[i], written_pins3[i]);

    //Check edge weights
    auto org_netWeightsVal3 = org_netWeights3->get_vals();
    auto written_netWeightsVal3 = written_netWeights3->get_vals();
    for (int i = 0; i < org_n3; ++i)
        EXPECT_EQ(org_netWeightsVal3[i], written_netWeightsVal3[i]);

    //Check cell weights
    auto org_cellWeightsVal3 = org_cellWeights3->get_vals();
    auto written_cellWeightsVal3 = written_cellWeights3->get_vals(); 

    for (int i = 0; i < org_vertex_size3; ++i)
        EXPECT_EQ(org_cellWeightsVal3[i], written_cellWeightsVal3[i]);

    // Check xnets
    for (int i = 0; i<org_vertex_size3+1;++i)
         EXPECT_EQ(org_xnets3[i], written_xnets3[i]);
    
    // Check cells
    for (int i = 0; i<org_m3;++i)
         EXPECT_EQ(org_cells3[i], written_cells3[i]);

    // TEST4

    //Write the hypergraph data with both vertices and nets weight to a file
    std::ofstream ofs4("HyperGraph_With_Cell_and_Nets_Weight.hypeg");
    ofs4 << hypergraph_4;
    ofs4.close();

    //Read the original hypergraph4 from data
    sparsebase::io::PatohReader<int,int,int> ReadHyperGraph4("HyperGraph_With_Cell_and_Nets_Weight.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* org_hypergraph4 = ReadHyperGraph4.ReadHyperGraph();
    sparsebase::format::Format *org_con4 = org_hypergraph4->get_connectivity();
    int org_n4 = org_hypergraph4->n_;
    int org_m4 = org_hypergraph4->m_;
    int org_constraint_num4 = org_hypergraph4->constraint_num_;
    int org_base_type4 = org_hypergraph4->base_type_;
    int org_vertex_size4 = org_con4->get_dimensions()[1];
    auto org_xpins4 = org_con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto org_pins4 = org_con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto org_xpin_val_arr4 = org_con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto org_xNetsCSR4 = org_hypergraph4->xNetCSR_;
    auto org_xnets4 = org_xNetsCSR4->get_row_ptr();
    auto org_cells4 = org_xNetsCSR4->get_col();
    auto org_xnet_val_arr4 = org_xNetsCSR4->get_vals();
    auto org_netWeights4 = org_hypergraph4->netWeights_;
    auto org_cellWeights4 = org_hypergraph4->cellWeights_;

    //Write the original graph4
    sparsebase::io::PatohWriter<int,int,int> WriteHyperGraph4("org_hypergraph4.hypeg",false,true,true);
    WriteHyperGraph4.WriteHyperGraph(org_hypergraph4);

    //Read the output of writer
    sparsebase::io::PatohReader<int,int,int> Read_Written_HyperGraph4("org_hypergraph4.hypeg");
    sparsebase::object::HyperGraph<int,int,int>* written_hypergraph4 = Read_Written_HyperGraph4.ReadHyperGraph();
    sparsebase::format::Format *written_con4 = written_hypergraph4->get_connectivity();

    int written_n4 = written_hypergraph4->n_;
    int written_m4 = written_hypergraph4->m_;
    int written_constraint_num4 = written_hypergraph4->constraint_num_;
    int written_base_type4 = written_hypergraph4->base_type_;
    int written_vertex_size4 = written_con4->get_dimensions()[1];
    auto written_xpins4 = written_con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto written_pins4 = written_con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto written_xpin_val_arr4 = written_con4->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();
    auto written_xNetsCSR4 = written_hypergraph4->xNetCSR_;
    auto written_xnets4 = written_xNetsCSR4->get_row_ptr();
    auto written_cells4 = written_xNetsCSR4->get_col();
    auto written_xnet_val_arr4 = written_xNetsCSR4->get_vals();
    auto written_netWeights4 = written_hypergraph4->netWeights_;
    auto written_cellWeights4 = written_hypergraph4->cellWeights_;

    //Compare the original hypergraph 4 with the written version
    //Check the dimensions
    EXPECT_EQ(org_n4,written_n4);
    EXPECT_EQ(org_vertex_size4, written_vertex_size4);
    EXPECT_EQ(org_m4,written_m4);
    EXPECT_EQ(org_constraint_num4,written_constraint_num4);
    EXPECT_EQ(org_base_type4,written_base_type4);

    EXPECT_EQ(written_xpin_val_arr4, nullptr);
    EXPECT_EQ(written_xnet_val_arr4, nullptr);

     //Check xpins
    for (int i = 0; i < org_n4+1; ++i)
        EXPECT_EQ(org_xpins4[i], written_xpins4[i]);

    //Check pins
    for (int i = 0; i < org_m4; ++i)
        EXPECT_EQ(org_pins4[i], written_pins4[i]);

    //Check edge weights
    auto org_netWeightsVal4 = org_netWeights4->get_vals();
    auto written_netWeightsVal4 = written_netWeights4->get_vals();
    for (int i = 0; i < org_n4; ++i)
        EXPECT_EQ(org_netWeightsVal4[i], written_netWeightsVal4[i]);

    //Check cell weights
    auto org_cellWeightsVal4 = org_cellWeights4->get_vals();
    auto written_cellWeightsVal4 = written_cellWeights4->get_vals(); 

    for (int i = 0; i < org_vertex_size4; ++i)
        EXPECT_EQ(org_cellWeightsVal4[i], written_cellWeightsVal4[i]);

    // Check xnets
    for (int i = 0; i<org_vertex_size4+1;++i)
         EXPECT_EQ(org_xnets4[i], written_xnets4[i]);
    
    // Check cells
    for (int i = 0; i<org_m4;++i)
         EXPECT_EQ(org_cells4[i], written_cells4[i]);

}
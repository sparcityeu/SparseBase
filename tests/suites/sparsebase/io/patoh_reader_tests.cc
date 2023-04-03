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
    int vertex_size = con->get_dimensions()[1];
    auto xpins = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_row_ptr();
    auto pins = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_col();
    auto netWeights = con->AsAbsolute<sparsebase::format::CSR<int,int,int>>()->get_vals();

    //Check the dimensions
    EXPECT_EQ(n_,hypergraph1_n);
    EXPECT_EQ(vertex_size, hypergraph1_vertex_size);
    EXPECT_EQ(m_,hypergraph1_m);
    EXPECT_NE(xpins, nullptr);
    EXPECT_NE(pins, nullptr);
    EXPECT_NE(netWeights, nullptr);

    //Check xpins
    for (int i = 0; i < n_; ++i)
        EXPECT_EQ(hypergraph1_xpins[i], xpins[i]);

    //Check pins
    for (int i = 0; i < vertex_size; ++i)
        EXPECT_EQ(hypergraph1_pins[i], pins[i]);

    //Check edge weights
    for (int i = 0; i < m_; ++i)
    EXPECT_EQ(hypergraph1_netWeights[i], netWeights[i]);

}
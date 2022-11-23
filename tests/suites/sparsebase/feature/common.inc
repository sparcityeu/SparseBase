using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = float;
using feature_type = float;

using namespace std;

using namespace sparsebase;
using namespace format;
using namespace preprocess;
using namespace feature;
using namespace context;

class COOMock : public ::testing::Test {
 protected:
  COO<vertex_type, edge_type, value_type> *coo;
  vertex_type rows[6] = {0, 0, 1, 1, 2, 3};
  edge_type cols[6] = {0, 1, 1, 2, 3, 3};
  value_type vals[6] = {10, 20, 30, 40, 50, 60};

  void SetUp() override {
    coo = new COO<vertex_type, edge_type, value_type>(6, 6, 6, rows, cols, vals,
                                                      kNotOwned);
  }
  void TearDown() override { delete coo; }
};

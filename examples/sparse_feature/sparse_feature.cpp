#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_object.h"
#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_exception.h"
#include "sparsebase/sparse_feature.h"

using namespace std;
using namespace sparsebase;


template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

int main(int argc, char * argv[]){
  /*
  if (argc < 2){
    cout << "Usage: ./sparse_feature <matrix_market_format>\n";
    cout << "Hint: You can use the matrix market file: examples/data/ash958.mtx\n";
    return 1;
  }
  */
  {
    unsigned int row_ptr[4] = {0, 2, 3, 4};
    unsigned int col[4] = {1, 2, 0, 0};
    float val[4] = {1.0, 2.0, 3.0, 4.0};
    CSR<unsigned int, unsigned int, float> * csr = new CSR<unsigned int, unsigned int, float>(3, 3, row_ptr, col, val);
    auto format = csr->get_format();
    auto dimensions = csr->get_dimensions();
    auto row_ptr2 = csr->get_row_ptr();
    auto col2 = csr->get_col();
    auto vals = csr->get_vals();
    cout << "Format: " << format << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for(int i = 0; i < dimensions.size(); i++){
      cout << "Dim " << i << " size " << dimensions[i] << endl; 
    }

    SparseFeature<unsigned int, unsigned int, float> * sparse_feature = new SparseFeature<unsigned int, unsigned int, float>();
    sparse_feature->Extract(csr);
    std::vector<Feature> features = sparse_feature->ListFeatures();
    cout << "Sparse Features: " << endl;
    for (auto feature : features) {
      std::string name = sparse_feature->GetFeatureName(feature);
      unsigned int order = sparse_feature->GetFeature(feature)->GetOrder();
      std::vector<unsigned int> dimensions = sparse_feature->GetFeature(feature)->GetDimension();
      std::string dimensions_str;
      for (auto dimension : dimensions) {
        dimensions_str += string_format("%u", dimension);
      }
      std::any value = *(sparse_feature->GetFeature(feature)->Value());
      cout << "  " << name  << " " << "{" << order << "}" << "["<< dimensions_str << "]" << " = " << (value.has_value() ? string_format("%f", std::any_cast<float>(value)) : "?") << endl;
    }
  }

  cout << endl;
  cout <<  "************************" << endl;
  cout <<  "************************" << endl;
  cout << endl;

  /*
  {
    string file_name = argv[1];
    MTXReader<unsigned int, unsigned int, void> reader(file_name);
    COO<unsigned int, unsigned int, void> * coo = reader.ReadCOO();
    auto format = coo->get_format();
    auto dimensions = coo->get_dimensions();
    auto coo_col = coo->get_col();
    try{ //if the data member is invalid, sparse format throws an exception
      auto coo_row_ptr = coo->get_row_ptr();
    }
    catch(InvalidDataMember& ex){
      cout << ex.what() << endl;
    }
    auto coo_row = coo->get_row();
    auto coo_vals = coo->get_vals();
    cout << "Format: " << coo->get_format() << endl;
    cout << "# of dimensions: " << dimensions.size() << endl;
    for(int i = 0; i < dimensions.size(); i++){
      cout << "Dim " << i << " size " << dimensions[i] << endl; 
    }
  }
  */

  return 0;
}

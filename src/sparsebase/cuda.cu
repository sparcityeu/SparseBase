#include <iostream>
using namespace std;
__global__ void kernel(){}

int main(){

  if (std::is_same_v<void, void>) cout << "endl\n";

return 0;
}

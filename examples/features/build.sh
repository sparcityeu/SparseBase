cd ../..
mkdir build
cd build
rm -r *
pwd
cmake -D_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DCUDA=${CUDA} ..
make
cmake --install .
make

cd ../examples/features/
g++ -std=c++17 feature.cc
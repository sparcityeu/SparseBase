#! /bin/bash
find ${1}/src/ -iname "*.h" -o -iname "*.cc" -o -iname "*.cuh" -o -iname "*.cu" | xargs clang-format -i 
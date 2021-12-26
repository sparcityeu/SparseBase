#! /bin/bash
find ${1}/sparsebase/ -iname *.hpp -o -iname *.cpp | xargs clang-format -i 
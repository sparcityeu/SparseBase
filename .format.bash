#! /bin/bash
find ${1}/sparsebase/ -iname *.h -o -iname *.cc | xargs clang-format -i 
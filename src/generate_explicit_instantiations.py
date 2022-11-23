# This script is intended to be executed by CMake directly
# Manually executing this script should not be necessary other than debugging purposes

import sys
import os
import argparse
import itertools
import json

parser = argparse.ArgumentParser(description="Generate explicit instantiations of SparseBase template classes")
parser.add_argument('--id-types', nargs='+', type=str, help= "C++ data types used for variables storing IDs")
parser.add_argument('--nnz-types', nargs='+', type=str, help= "C++ data types used for variables storing numbers of non-zeros")
parser.add_argument('--value-types', nargs='+', type=str, help= "C++ data types used for variables storing values inside formats")
parser.add_argument('--float-types', nargs='+', type=str, help= "C++ data types used for variables storing floating point numbers")
parser.add_argument('--output-folder', type=str, help= "Path to output folder to store resultant files")
parser.add_argument('--dry-run', action='store_true', help= "Will not write the files to disk, and will write them to stdout instead")

parser.add_argument('--class-list', type=str, help= "Path to class instantiation json list")

parser.add_argument('--pigo', type=str, help= "Use pigo (OFF/ON).")
parser.add_argument('--cuda', type=str, help= "Use CUDA (OFF/ON).")
args = parser.parse_args()

id_types = ' '.join(args.id_types).split(',')
nnz_types = ' '.join(args.nnz_types).split(',')
value_types = ' '.join(args.value_types).split(',')
float_types = ' '.join(args.float_types).split(',')

type_to_idx_map = {"$id_type": 0, "$nnz_type": 1, "$value_type": 2, "$float_type": 3}

output_folder = args.output_folder
class_list = args.class_list
cuda_output_folder = os.path.join(output_folder, 'cuda')
dry_run = args.dry_run

os.makedirs(output_folder, exist_ok=True)
os.makedirs(cuda_output_folder, exist_ok=True)

comb_types = list(itertools.product(id_types, nnz_types, value_types, float_types))
already_added = set()

def reset_file(filename, folder=output_folder):
    path = os.path.join(folder, filename)
    out = open(path, "w+")
    out.write("// Automatically generated by " + os.path.basename(__file__) + "\n\n")
    out.write("#include <cstdint>\n\n")
    out.close()


def gen_inst(template, filename, ifdef=None, folder=output_folder, exceptions=None):
    local_comb_types = comb_types
    if exceptions is not None:
        for key, exception_types in exceptions.items():
            if key not in type_to_idx_map.keys():
                raise "Illegal type template"
            local_comb_types = [types for types in local_comb_types if types[type_to_idx_map[key]] not in
                                exception_types]
    out = sys.stdout
    if not dry_run:
        path = os.path.join(folder, filename)
        out = open(path, "a")

    out.write("// Generated from: " + template + "\n")

    if ifdef is not None:
        out.write("#ifdef " + ifdef + "\n")

    for types in local_comb_types:
        inst = template.replace("$id_type", types[0])
        inst = inst.replace("$nnz_type", types[1])
        inst = inst.replace("$value_type", types[2])
        inst = inst.replace("$float_type", types[3])

        key = folder + ":" + filename + ":" + inst

        if key not in already_added:
            already_added.add(key)
            out.write("template  " + inst + ";\n")


    if ifdef is not None:
        out.write("#endif\n")

    out.write("\n")

    if not dry_run:
        out.close()


#reading class registration file
with open(class_list, 'r') as read_file:
    data = json.load(read_file)

opened_files = {}
class_instantiation_list = data.get("classes")
for each_class in class_instantiation_list:

    file_name = each_class.get("filename")
    if each_class.get("folder"):
        store_at = os.path.join(output_folder, each_class.get("folder"))
    else:
        store_at = output_folder

    if not (file_name, store_at) in opened_files:
        opened_files[(file_name, store_at)] = True
        reset_file(file_name, store_at)


    gen_inst(
        template=each_class.get("template"),
        filename=file_name,
        ifdef=each_class.get("ifdef"),
        folder=store_at,
        exceptions=each_class.get("exceptions")
        )
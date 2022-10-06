# This script is intended to be executed by CMake directly
# Manually executing this script should not be necessary other than debugging purposes

import sys
import os
import argparse
import itertools

parser = argparse.ArgumentParser(description="Generate explicit instantiations of SparseBase template classes")
parser.add_argument('--id-types', nargs='+', type=str, help= "C++ data types used for variables storing IDs")
parser.add_argument('--nnz-types', nargs='+', type=str, help= "C++ data types used for variables storing numbers of non-zeros")
parser.add_argument('--value-types', nargs='+', type=str, help= "C++ data types used for variables storing values inside formats")
parser.add_argument('--float-types', nargs='+', type=str, help= "C++ data types used for variables storing floating point numbers")
parser.add_argument('--output-folder', type=str, help= "Path to output folder to store resultant files")
parser.add_argument('--dry-run', action='store_true', help= "Will not write the files to disk, and will write them to stdout instead")

parser.add_argument('--pigo', type=str, help= "Use pigo (OFF/ON).")
parser.add_argument('--cuda', type=str, help= "Use CUDA (OFF/ON).")
args = parser.parse_args()

id_types = ' '.join(args.id_types).split(',')
nnz_types = ' '.join(args.nnz_types).split(',')
value_types = ' '.join(args.value_types).split(',')
float_types = ' '.join(args.float_types).split(',')

type_to_idx_map = {"$id_type": 0, "$nnz_type": 1, "$value_type": 2, "$float_type": 3}

output_folder = args.output_folder
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
    out.close()


def gen_inst(template, filename, is_class=True, ifdef=None, folder=output_folder, exceptions=None):
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
            if is_class:
                out.write("template class " + inst + ";\n")
            else:
                out.write("template  " + inst + ";\n")


    if ifdef is not None:
        out.write("#endif\n")

    out.write("\n")

    if not dry_run:
        out.close()


# Format
reset_file("format.inc")
gen_inst("CSR<$id_type, $nnz_type, $value_type>", "format.inc")
gen_inst("CSC<$id_type, $nnz_type, $value_type>", "format.inc")
gen_inst("COO<$id_type, $nnz_type, $value_type>", "format.inc")
gen_inst("Array<$value_type>", "format.inc", exceptions={"$value_type": ['void']})
gen_inst("Array<$nnz_type>", "format.inc", exceptions={"$nnz_type": ['void']})
gen_inst("Array<$id_type>", "format.inc", exceptions={"$id_type": ['void']})

# Format (CUDA)
reset_file("format.inc", cuda_output_folder)
gen_inst("CUDACSR<$id_type, $nnz_type, $value_type>", "format.inc",
         ifdef="USE_CUDA", folder=cuda_output_folder)
gen_inst("CUDAArray<$value_type>", "format.inc",
         ifdef="USE_CUDA", folder=cuda_output_folder, exceptions={"$value_type": ['void']})
gen_inst("CUDAArray<$id_type>", "format.inc",
         ifdef="USE_CUDA", folder=cuda_output_folder, exceptions={"$value_type": ['void']})
gen_inst("CUDAArray<$nnz_type>", "format.inc",
         ifdef="USE_CUDA", folder=cuda_output_folder, exceptions={"$value_type": ['void']})


# Object
reset_file("object.inc")
gen_inst("AbstractObject<$id_type, $nnz_type, $value_type>", "object.inc")
gen_inst("Graph<$id_type, $nnz_type, $value_type>", "object.inc")

# Converter
reset_file("converter.inc")
gen_inst("ConverterOrderOne<$value_type>", "converter.inc", exceptions={"$value_type": ['void']})
gen_inst("ConverterOrderOne<$id_type>", "converter.inc", exceptions={"$id_type": ['void']})
gen_inst("ConverterOrderOne<$nnz_type>", "converter.inc", exceptions={"$nnz_type": ['void']})
gen_inst("ConverterOrderTwo<$id_type, $nnz_type, $value_type>", "converter.inc")

# Converter (CUDA)
reset_file("converter.inc", folder=cuda_output_folder)
return_type = "format::Format * "
params = "(format::Format *source, context::Context*context)"
order_two_functions = ['CsrCUDACsrConditionalFunction',
                       'CUDACsrCsrConditionalFunction',
                       'CUDACsrCUDACsrConditionalFunction']
order_one_functions = ['CUDAArrayArrayConditionalFunction',
                       'ArrayCUDAArrayConditionalFunction']

for func in order_two_functions:
    gen_inst(return_type + func + "<$id_type, $nnz_type, $value_type>" + params, "converter.inc", is_class=False,
             ifdef="USE_CUDA", folder=cuda_output_folder)

for func in order_one_functions:
    gen_inst(return_type + func + "<$id_type>" + params, "converter.inc", is_class=False,
         ifdef="USE_CUDA", folder=cuda_output_folder, exceptions={"$id_type": ['void']})
    gen_inst(return_type + func + "<$nnz_type>" + params, "converter.inc", is_class=False,
         ifdef="USE_CUDA", folder=cuda_output_folder, exceptions={"$nnz_type": ['void']})
    gen_inst(return_type + func + "<$value_type>" + params, "converter.inc", is_class=False,
         ifdef="USE_CUDA", folder=cuda_output_folder, exceptions={"$value_type": ['void']})



# Reader
reset_file("reader.inc")
gen_inst("EdgeListReader<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("MTXReader<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("PigoEdgeListReader<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("PigoMTXReader<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("BinaryReaderOrderTwo<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("BinaryReaderOrderOne<$id_type>", "reader.inc", exceptions={"$id_type": ['void']})
gen_inst("BinaryReaderOrderOne<$nnz_type>", "reader.inc", exceptions={"$nnz_type": ['void']})
gen_inst("BinaryReaderOrderOne<$value_type>", "reader.inc", exceptions={"$value_type": ['void']})

# Writer
reset_file("writer.inc")
gen_inst("BinaryWriterOrderOne<$id_type>", "writer.inc", exceptions={"$id_type": ['void']})
gen_inst("BinaryWriterOrderOne<$nnz_type>", "writer.inc", exceptions={"$nnz_type": ['void']})
gen_inst("BinaryWriterOrderOne<$value_type>", "writer.inc", exceptions={"$value_type": ['void']})
gen_inst("BinaryWriterOrderTwo<$id_type, $nnz_type, $value_type>", "writer.inc")

# Feature
reset_file("feature.inc")
gen_inst("FeatureExtractor<$id_type, $nnz_type, $value_type, $float_type>", "feature.inc")

# Preprocess
reset_file("preprocess.inc")
gen_inst("Degrees<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("RCMReorder<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("DegreeReorder<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("GenericReorder<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("PermuteOrderTwo<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("PermuteOrderOne<$id_type, $value_type>", "preprocess.inc", exceptions={"$value_type": ['void']})
gen_inst("TransformPreprocessType<format::FormatOrderTwo<$id_type, $nnz_type, $value_type>, format::FormatOrderTwo<$id_type, $nnz_type, $value_type>>", "preprocess.inc")
gen_inst("TransformPreprocessType<format::FormatOrderOne<$value_type>, format::FormatOrderOne<$value_type>>", "preprocess.inc")
gen_inst("TransformPreprocessType<format::FormatOrderOne<$id_type>, format::FormatOrderOne<$id_type>>", "preprocess.inc")
gen_inst("TransformPreprocessType<format::FormatOrderOne<$nnz_type>, format::FormatOrderOne<$nnz_type>>", "preprocess.inc")
gen_inst("DegreeDistribution<$id_type, $nnz_type, $value_type, $float_type>", "preprocess.inc")
gen_inst("Degrees_DegreeDistribution<$id_type, $nnz_type, $value_type, $float_type>", "preprocess.inc")
gen_inst("JaccardWeights<$id_type, $nnz_type, $value_type, $float_type>", "preprocess.inc")
gen_inst("MetisPartition<$id_type, $nnz_type, $value_type>", "preprocess.inc", ifdef="USE_METIS")
gen_inst("ConverterMixin<PreprocessType>", "preprocess.inc")
gen_inst("FunctionMatcherMixin<format::Format*>", "preprocess.inc")
gen_inst("FunctionMatcherMixin<$id_type*>", "preprocess.inc")
gen_inst("FunctionMatcherMixin<$id_type>", "preprocess.inc")
gen_inst("FunctionMatcherMixin<$id_type*, ConverterMixin<ExtractableType>>", "preprocess.inc")
gen_inst("FunctionMatcherMixin<$id_type, ConverterMixin<ExtractableType>>", "preprocess.inc")
gen_inst("GenericPreprocessType<$id_type>", "preprocess.inc")
gen_inst("ReorderPreprocessType<$id_type>", "preprocess.inc")
gen_inst("PartitionPreprocessType<$id_type>", "preprocess.inc")

# Preprocess (CUDA)
reset_file("preprocess.inc", cuda_output_folder)
gen_inst("format::cuda::CUDAArray<$float_type>* "
         + "RunJaccardKernel<$id_type, $nnz_type, $value_type, $float_type>(format::cuda::CUDACSR<$id_type, $nnz_type, $value_type>*)",
         "preprocess.inc",
         is_class=False,
         ifdef="USE_CUDA",
         folder=cuda_output_folder)

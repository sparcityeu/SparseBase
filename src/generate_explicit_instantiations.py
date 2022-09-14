from sys import stdout
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
output_folder = args.output_folder
dry_run = args.dry_run
#external_output_folder = os.path.join(output_folder, 'external')
#cuda_output_folder = os.path.join(external_output_folder, 'cuda')


os.makedirs(output_folder, exist_ok=True)

comb_types = list(itertools.product(id_types, nnz_types, value_types, float_types))
already_added = set()

def gen_inst(template, filename, ifdef=None, folder=output_folder):

    out = stdout
    if not dry_run:
        path = os.path.join(folder, filename)
        out = open(path, "a")

    out.write("// Generated from: " + template + "\n")

    if ifdef is not None:
        out.write("#ifdef " + ifdef + "\n")

    for types in comb_types:
        inst = template.replace("$id_type", types[0])
        inst = inst.replace("$nnz_type", types[1])
        inst = inst.replace("$value_type", types[2])
        inst = inst.replace("$float_type", types[3])

        if inst+filename not in already_added:
            already_added.add(inst+filename)
            out.write("template class " + inst + ";\n")

    if ifdef is not None:
        out.write("#endif\n")

    out.write("\n")

    if not dry_run:
        out.close()

# Format
gen_inst("CSR<$id_type, $nnz_type, $value_type>", "format.inc")
gen_inst("COO<$id_type, $nnz_type, $value_type>", "format.inc")
gen_inst("Array<$value_type>", "format.inc")
gen_inst("Array<$nnz_type>", "format.inc")
gen_inst("Array<$id_type>", "format.inc")

# Object
gen_inst("AbstractObject<$id_type, $nnz_type, $value_type>", "object.inc")
gen_inst("Graph<$id_type, $nnz_type, $value_type>", "object.inc")

# Converter
gen_inst("ConverterOrderOne<$value_type>", "converter.inc")
gen_inst("ConverterOrderOne<$id_type>", "converter.inc")
gen_inst("ConverterOrderOne<$nnz_type>", "converter.inc")
gen_inst("ConverterOrderTwo<$id_type, $nnz_type, $value_type>", "converter.inc")

# Reader
gen_inst("EdgeListReader<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("MTXReader<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("PigoEdgeListReader<$id_type, $nnz_type, $value_type>", "reader.inc", ifdef="USE_PIGO")
gen_inst("PigoMTXReader<$id_type, $nnz_type, $value_type>", "reader.inc", ifdef="USE_PIGO")
gen_inst("BinaryReaderOrderTwo<$id_type, $nnz_type, $value_type>", "reader.inc")
gen_inst("BinaryReaderOrderOne<$id_type>", "reader.inc")
gen_inst("BinaryReaderOrderOne<$nnz_type>", "reader.inc")
gen_inst("BinaryReaderOrderOne<$value_type>", "reader.inc")

# Writer
gen_inst("BinaryWriterOrderOne<$id_type>", "writer.inc")
gen_inst("BinaryWriterOrderOne<$nnz_type>", "writer.inc")
gen_inst("BinaryWriterOrderOne<$value_type>", "writer.inc")
gen_inst("BinaryReaderOrderTwo<$id_type, $nnz_type, $value_type>", "writer.inc")

# Feature
gen_inst("FeatureExtractor<$id_type, $nnz_type, $value_type, $float_type", "feature.inc")

# Preprocessors
gen_inst("Degrees<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("RCMReorder<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("DegreeReorder<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("GenericReorder<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("Transform<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("TransformPreprocessType<$id_type, $nnz_type, $value_type>", "preprocess.inc")
gen_inst("DegreeDistribution<$id_type, $nnz_type, $value_type, $float_type", "feature.inc")
gen_inst("Degrees_DegreeDistribution<$id_type, $nnz_type, $value_type, $float_type", "feature.inc")
gen_inst("JaccardWeights<$id_type, $nnz_type, $value_type, $float_type", "feature.inc")
gen_inst("MetisPartition<$id_type, $nnz_type, $value_type>", "preprocess.inc", ifdef="USE_METIS")

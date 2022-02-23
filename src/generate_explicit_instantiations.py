from sys import argv
import os
import argparse

parser = argparse.ArgumentParser(description="Generate explicit instantiations of SparseBase template classes")
parser.add_argument('--id-types', nargs='+', type=str, help= "C++ data types used for variables storing IDs")
parser.add_argument('--nnz-types', nargs='+', type=str, help= "C++ data types used for variables storing numbers of non-zeros")
parser.add_argument('--value-types', nargs='+', type=str, help= "C++ data types used for variables storing values inside formats")
parser.add_argument('--float-types', nargs='+', type=str, help= "C++ data types used for variables storing floating point numbers")
parser.add_argument('--output-folder', type=str, help= "Path to output folder to store resultant files")
args = parser.parse_args()

'''
Global variables defining data types
'''
vertex_types = ' '.join(args.id_types).split(',')
nnz_types = ' '.join(args.nnz_types).split(',')
value_types = ' '.join(args.value_types).split(',')
float_types = ' '.join(args.float_types).split(',')
output_folder = args.output_folder

PREFIX = "template class "
temp_suffix = '.tmp'

## Utility function that prints explicit instantiations of classes that require exactly three template parameters: IDType, NNZType, and ValueType
# @param implementations: a list of strings corresponding to class names
# @param out_stream: an output stream to which the instantiations will be printed
def print_implementations(implementations, out_stream):
    for implementation_class in implementations:
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    out_stream.write(PREFIX+implementation_class+'<'+vertex_type+', '+nnz_type+', '+value_type+'>;\n')
        out_stream.write('\n\n\n')


## Prints explicit template instantiations for the preprocess file
def preprocess_cc_init():
    filename = os.path.join(output_folder,'preprocess.inc'+temp_suffix)
    with open(filename, 'w') as fout:
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for preprocess_return_type in [vertex_type+'*', 'Format*']: 
                        fout.write(PREFIX+"FunctionMatcherMixin<"+vertex_type+", "+nnz_type+", "+value_type+", "+preprocess_return_type+", "+"ConverterMixin<PreprocessType, "+vertex_type+", "+nnz_type+", "+value_type+">>;\n")
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for dist_type in float_types:
                        fout.write(PREFIX+"DegreeDistribution<"+vertex_type+", "+nnz_type+", "+value_type+", "+dist_type+">;\n")
        print_implementations(['ReorderPreprocessType', 'GenericReorder', 'DegreeReorder', 'RCMReorder', 'TransformPreprocessType', 'Transform'], fout)
    return filename

## Prints explicit template instantiations for the converter file
def converter_cc_init():
    filename = os.path.join(output_folder,'converter.inc'+temp_suffix)
    with open(filename, 'w') as fout:
        print_implementations(['Converter', 'CsrCooFunctor', 'CooCsrFunctor'], fout)
    return filename

## Prints explicit template instantiations for the object file
def object_cc_init():
    filename = os.path.join(output_folder,'object.inc'+temp_suffix)
    with open(filename, 'w') as fout:
        print_implementations(['AbstractObject', 'Graph'], fout)
    return filename

## Prints explicit template instantiations for the format file
def format_cc_init():
    filename = os.path.join(output_folder,'format.inc'+temp_suffix)
    with open(filename, 'w') as fout:
        print_implementations(['CSR', 'COO'], fout)
    return filename

## Prints explicit template instantiations for the reader file
def reader_cc_init():
    filename = os.path.join(output_folder,'reader.inc'+temp_suffix)
    with open(filename, 'w') as fout:
        print_implementations(['MTXReader', 'UedgelistReader'], fout)
    return filename

## Create the output folder if it doesn't already exist
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
## Create temporary files containing the explicit instantiations
files_created = []
files_created.append(preprocess_cc_init())
files_created.append(converter_cc_init())
files_created.append(reader_cc_init())
files_created.append(format_cc_init())
files_created.append(object_cc_init())

## Check whether the newly created folder is different from the existing one, and if it is, replace it
for file_created in files_created:
    existing_file = file_created[:file_created.rfind(temp_suffix)]
    replace_old_file = True
    if os.path.isfile(existing_file):
        with open(file_created, 'r') as new_file:
            with open(existing_file, 'r') as old_file:
                old_line, new_line = None,None
                while old_line == new_line and old_line != '':
                    old_line = old_file.readline()
                    new_line = new_file.readline()
                if old_line == new_line:
                    replace_old_file = False 
    if replace_old_file:
        os.rename(file_created, existing_file)
        print("Updating explicit instantiation file:", existing_file)
    else:
        os.remove(file_created)
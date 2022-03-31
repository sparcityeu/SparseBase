from sys import stdout
import os
import argparse

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

'''
Global variables defining data types
'''
vertex_types = ' '.join(args.id_types).split(',')
nnz_types = ' '.join(args.nnz_types).split(',')
value_types = ' '.join(args.value_types).split(',')
float_types = ' '.join(args.float_types).split(',')
output_folder = args.output_folder
external_output_folder = os.path.join(output_folder, 'external')
use_pigo = True if args.pigo == 'ON' else False

use_cuda = True if args.cuda == 'ON' else False
cuda_output_folder = os.path.join(external_output_folder, 'cuda')

dry_run = args.dry_run

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

## An class for an objec that writes explicit initializations
# 
# This class manages opening the output stream and creating the temporary filename
class explicit_initialization:
    ## initializes the object
    # @param filename: the path to which the output is stored
    # @param dry_run: if true, prints the explicit initializations to stdout 
    def __init__(self, filename, dry_run=False):
        self.filename = filename

        if not dry_run:
            self.out_stream = open(self.get_temp_filename(), 'w')
        else:
            self.out_stream = stdout

    ## Returns the name of the temp file with the explicit initializations
    def get_temp_filename(self):
        return self.filename+temp_suffix

    ## Returns the name of the final file with the explicit initializations
    def get_filename(self):
        return self.filename

    ## If this isn't a dry run, closes the stream to the file
    def close(self):
        if not dry_run:
            self.out_stream.close()

class converter_cuda_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'converter.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)

    ## Prints explicit template instantiations for the preprocess file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        order_two_functions = ['CsrCUDACsrConditionalFunction', 'CUDACsrCsrConditionalFunction', 'CUDACsrCUDACsrConditionalFunction']
        order_one_functions = ['CUDAArrayArrayConditionalFunction', 'ArrayCUDAArrayConditionalFunction']
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for function in order_two_functions: 
                        self.out_stream.write("template sparsebase::format::Format * "+function+"<"+vertex_type+", "+nnz_type+", "+value_type+">(sparsebase::format::Format *source, sparsebase::context::Context*context);\n")
        for value_type in value_types:
            for function in order_one_functions: 
                self.out_stream.write("template sparsebase::format::Format * "+function+"<"+value_type+">(sparsebase::format::Format *source, sparsebase::context::Context*context);\n")

class preprocess_cuda_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'preprocess.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)

    ## Prints explicit template instantiations for the preprocess file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        order_two_functions = [('JaccardWeights', ['GetJaccardWeightCUDACSR'])]
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for float_type in float_types:
                        for prep_class, funcs in order_two_functions: 
                            for function in funcs:
                                types = "<"+vertex_type+", "+nnz_type+", "+value_type+", "+float_type+">"
                                self.out_stream.write("template sparsebase::format::Format * preprocess::"+prep_class+types+'::'+function+"(std::vector<sparsebase::format::Format *> formats, sparsebase::preprocess::PreprocessParams * params);\n")

class format_cuda_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'format.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the format file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        single_order_classes = ['CUDAArray']
        for value_type in value_types:
            for c in single_order_classes:
                self.out_stream.write(PREFIX+c+"<"+value_type+">;\n")
        self.out_stream.write('\n\n')
        print_implementations(['CUDACSR'], self.out_stream)
class preprocess_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'preprocess.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)

    ## Prints explicit template instantiations for the preprocess file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        self.out_stream.write(PREFIX+"FunctionMatcherMixin<"+ 'Format*'+", "+"ConverterMixin<PreprocessType>>;\n")
        for vertex_type in vertex_types:
            for preprocess_return_type in [vertex_type+'*']:
                self.out_stream.write(PREFIX+"FunctionMatcherMixin<"+preprocess_return_type+", "+"ConverterMixin<PreprocessType>>;\n")
                self.out_stream.write(PREFIX+"FunctionMatcherMixin<"+preprocess_return_type+", "+"ConverterMixin<ExtractableType>>;\n")
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for dist_type in float_types:
                        self.out_stream.write(PREFIX+"DegreeDistribution<"+vertex_type+", "+nnz_type+", "+value_type+", "+dist_type+">;\n")
                        self.out_stream.write(PREFIX+"Degrees_DegreeDistribution<"+vertex_type+", "+nnz_type+", "+value_type+", "+dist_type+">;\n")
                        self.out_stream.write(PREFIX+"JaccardWeights<"+vertex_type+", "+nnz_type+", "+value_type+", "+dist_type+">;\n")
        print_implementations(['ReorderPreprocessType', 'GenericReorder', 'DegreeReorder', 'RCMReorder', 'TransformPreprocessType', 'Transform'], self.out_stream)

class converter_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'converter.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)

    ## Prints explicit template instantiations for the converter file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        single_order_classes = ['OrderOneConverter']
        for value_type in value_types:
            for c in single_order_classes:
                self.out_stream.write(PREFIX+c+"<"+value_type+">;\n")
        print_implementations(['OrderTwoConverter'], self.out_stream)

class object_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'object.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the object file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        print_implementations(['AbstractObject', 'Graph'], self.out_stream)

class format_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'format.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the format file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        single_order_classes = ['Array']
        for value_type in value_types:
            for c in single_order_classes:
                self.out_stream.write(PREFIX+c+"<"+value_type+">;\n")
        self.out_stream.write('\n\n')
        print_implementations(['CSR', 'COO'], self.out_stream)

class reader_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'reader.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the reader file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        for value_type in value_types:
            self.out_stream.write("template class BinaryReaderOrderOne<" + value_type + ">;\n")
        print_implementations(['MTXReader', 'UedgelistReader', 'BinaryReaderOrderTwo'],
                              self.out_stream)

class writer_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'writer.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the reader file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        for value_type in value_types:
            self.out_stream.write("template class BinaryWriterOrderOne<" + value_type + ">;\n")
        print_implementations(['BinaryWriterOrderOne', 'BinaryWriterOrderTwo'], self.out_stream)


class pigo_reader_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'pigo.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the reader file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        print_implementations(['PigoMTXReader', 'PigoEdgeListReader'], self.out_stream)

class feature_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'feature.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the reader file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for dist_type in float_types:
                        self.out_stream.write(PREFIX+"FeatureExtractor<"+vertex_type+", "+nnz_type+", "+value_type+", "+dist_type+">;\n")

## Create the output folder if it doesn't already exist
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
if not os.path.isdir(external_output_folder):
    os.mkdir(external_output_folder)
if use_cuda:
    if not os.path.isdir(cuda_output_folder):
        os.mkdir(cuda_output_folder)

inits = []
inits.append(reader_init(output_folder, dry_run))
inits.append(format_init(output_folder, dry_run))
inits.append(converter_init(output_folder, dry_run))
inits.append(preprocess_init(output_folder, dry_run))
inits.append(object_init(output_folder, dry_run))
inits.append(feature_init(output_folder, dry_run))
inits.append(writer_init(output_folder, dry_run))
if use_cuda:
    inits.append(converter_cuda_init(cuda_output_folder, dry_run))
    inits.append(preprocess_cuda_init(cuda_output_folder, dry_run))
    inits.append(format_cuda_init(cuda_output_folder, dry_run))
if use_pigo:
    inits.append(pigo_reader_init(external_output_folder, dry_run))

## Create temporary files containing the explicit instantiations
for init_object in inits:
    init_object.run()
    init_object.close()

if not dry_run:
    ## Check whether the newly created folder is different from the existing one, and if it is, replace it
    for init_object in inits:
        file_created = init_object.get_temp_filename()
        existing_file = init_object.get_filename()
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
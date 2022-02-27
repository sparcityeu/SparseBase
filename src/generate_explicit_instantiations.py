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
args = parser.parse_args()

'''
Global variables defining data types
'''
vertex_types = ' '.join(args.id_types).split(',')
nnz_types = ' '.join(args.nnz_types).split(',')
value_types = ' '.join(args.value_types).split(',')
float_types = ' '.join(args.float_types).split(',')
output_folder = args.output_folder
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

class preprocess_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'preprocess.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)

    ## Prints explicit template instantiations for the preprocess file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for preprocess_return_type in [vertex_type+'*', 'Format*']: 
                        self.out_stream.write(PREFIX+"FunctionMatcherMixin<"+vertex_type+", "+nnz_type+", "+value_type+", "+preprocess_return_type+", "+"ConverterMixin<PreprocessType, "+vertex_type+", "+nnz_type+", "+value_type+">>;\n")
        for vertex_type in vertex_types:
            for nnz_type in nnz_types:
                for value_type in value_types:
                    for dist_type in float_types:
                        self.out_stream.write(PREFIX+"DegreeDistribution<"+vertex_type+", "+nnz_type+", "+value_type+", "+dist_type+">;\n")
        print_implementations(['ReorderPreprocessType', 'GenericReorder', 'DegreeReorder', 'RCMReorder', 'TransformPreprocessType', 'Transform'], self.out_stream)

class converter_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'converter.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)

    ## Prints explicit template instantiations for the converter file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        print_implementations(['Converter'], self.out_stream)

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
        print_implementations(['CSR', 'COO'], self.out_stream)

class reader_init(explicit_initialization):
    def __init__(self, folder, dry_run=False):
        self.source_filename = 'reader.inc'
        super().__init__(os.path.join(folder, self.source_filename), dry_run)
    ## Prints explicit template instantiations for the reader file
    def run(self):
        self.out_stream.write('// '+self.source_filename+'\n')
        print_implementations(['MTXReader', 'UedgelistReader'], self.out_stream)

## Create the output folder if it doesn't already exist
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

inits = []
inits.append(reader_init(output_folder, dry_run))
inits.append(format_init(output_folder, dry_run))
inits.append(converter_init(output_folder, dry_run))
inits.append(preprocess_init(output_folder, dry_run))
inits.append(object_init(output_folder, dry_run))
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
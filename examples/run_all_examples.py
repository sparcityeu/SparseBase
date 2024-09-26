import os
import subprocess as sp
import sys

# Each element should be a pair
# - First element is the directory and executable name (these two should be the same)
# - Second element is the input file (None if no input file is required)
# (The input file should be in the data directory at the same level as this file
examples = [
    ["custom_format", None],
    ["custom_order", "com-dblp.uedgelist"],
    ["degree_distribution", "com-dblp.uedgelist"],
    ["degree_order", "com-dblp.uedgelist"],
    ["format_conversion", None],
    ["rcm_order", "com-dblp.uedgelist"],
    ["sparse_feature", "ash958.mtx"],
    ["csr_coo", "ash958.mtx"],
    ["sparse_reader", "ash958.mtx"],
    ["tns_reader", "small.tns"],
]


def run_example(exe_filename, data_filename):
    dir_path = os.path.dirname(__file__)
    exe_path = os.path.join(dir_path, exe_filename, exe_filename)

    # Windows executables have ".exe" extension
    if os.name == "nt":
        exe_path += ".exe"

    if not os.path.exists(exe_path):
        print("Example is not built, skipping: ", exe_filename)
        return

    cmd = [exe_path]

    if data_filename is not None:
        data_path = os.path.join(dir_path, "data", data_filename)
        cmd.append(data_path)

    print("Running cmd=", cmd)
    result = sp.run(cmd)

    if result.returncode != 0:
        print("cmd=", cmd, "has failed with exit code=", result.returncode)
        sys.exit(1)


if __name__ == "__main__":
    for exe, data in examples:
        run_example(exe, data)

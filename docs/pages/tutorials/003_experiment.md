# Creating an SPMV Experiment

## Objective
Read matrix files from disk, apply reordering, run various SPMV kernels by using the experiment api. 

## Overview

In this tutorial, you will use SparseBase to do the following:

1. Read multiple matrix files (sparse matrices) from different file formats.
2. Reorder the rows of the matrices using Reverse Cuthill-McKee (RCM) reordering and permute the matrices using the obtained order.
3. Create multiple kernels that carry out sparse matrix / vector (SPMV) multiplication.
4. Use the `sparsebase::experiment::ConcreteExperiment` api to automate the above steps.

A single experiment runs all the preprocessings on all the loaded data. Then for all the generated data, i.e. loaded data times preprocessings, the experiment suite runs user defined kernels.

## Preliminaries
Start by navigating to the directory `tutorials/003_experiment/start_here/`. Open the file `tutorial_003.cc` using a code editor, and follow along with the tutorial. The file contains some boilerplate code that includes the appropriate headers, creates some type definitions, contains some functions for SPMV, and uses the `sparsebase` namespace.

The completed tutorial can be found in `tutorials/003_experiment/solved/solved.cc`. For this tutorial we will use the unordered edge list file `examples/data/com-dblp.uedgelist` and matrix market file `examples/data/ash958.mtx`. 

## Steps

### 1. Create an `experiment::ConcreteExperiment`.

In order to run a Sparsebase experiment first we need to create a `sparsebase::experiment::ConcreteExperiment`object.

```c++
  experiment::ConcreteExperiment exp;
```

Then we will add `DataLoader`, `Preprocess`, and `Kernel`functions, run the experiment, and analyze its results.

### 2. Add `DataLoaderFunction` to the Experiment.
Create data loaders for the two different file formats we are going to use. Also create and pass the vectors as file specific parameters. These vectors are going to be used in the calculation.

```c++
  // create vector for ash958.mtx and fill it with random floats
  auto v = new float[958];
  fill_r(v, 958);
  auto ash_v = Array<float>(958, v);
  // create vector for com-dblp.uedgelist and fill it with random floats
  v = new float[317080];
  fill_r(v, 317080);
  auto dblp_v = Array<float>(317080, v);
  // Add data loader for the first file
  vector<string> files = {argv[1]};
  exp.AddDataLoader(experiment::LoadCSR<MTXReader, row_type, nnz_type, value_type>, {make_pair(files, ash_v)});
  // add data loader for the second file
  files = {argv[2]};
  exp.AddDataLoader(experiment::LoadCSR<EdgeListReader, row_type, nnz_type, value_type>, {make_pair(files, dblp_v)});
```

Note that functions passed to `AddDataLoader` member function of the `experiment::Experiment` interface must follow the function definition `experiment::DataLoaderFunction`. 
In this example we are using the pre-written `experiment::LoadCSR`function by passing the correct file reader for the respective files.
We also provide the file/files to be read for each data loader, and file specific parameters that are going to be used throughout the experiment as pairs.

### 3. Add `PreprocessFunction` to the experiment.
Next, create two preprocessing functions, first use `experiment::Pass` function to run your kernels on the original data. Also add `experiment::Reorder` with the ordering of your choosing. In this case we add `RCMReorder`.
```c++
  // add dummy preprocessing to run kernels without reordering
  exp.AddPreprocess("original", experiment::Pass, {}); 
  RCMReorder<row_type, nnz_type, value_type>::ParamsType params = {};
  exp.AddPreprocess("RCM", experiment::Reorder<RCMReorder, CSR, CPUContext, row_type, nnz_type, value_type>, params);
```

As the first parameter of the `AddPreprocess` function we provide an identifier that is necessary to distinguish the results, auxiliary data, and runtimes generated as a result of the experiment.
The second parameter, preprocess function, must follow the function definition `experiment::PreprocessFunction`. Here we use the pre-written `experiment::Pass`to run kernels on the original data, and `experiment::Reorder`to reorder the matrices with RCM reordering.
Last parameter of this function is the preprocessing specific parameters. Here the preprocessing algorithms we wish to use do not use any parameters.


### 4. Add `KernelFunction` to the experiment.
After loading data and applying some preprocessing on it, now, we can add our own functions that we are actually interested in. In the file you can find two functions for SMPV; `spmv` and `spmv_par`. The first one is a single-threaded implementation and the second one uses `omp` to parallelization. 

Use the `AddKernel` function of the experiment api to add these functions to the pipeline.

```c++
  // add kernels that will carry out the SPMV
  // init random vals large enough for all the files and pass it as a kernel specific parameter
  auto vals = new float[2*1049866];
  fill_r(vals, 2*1049866);
  auto vals_v = Array<float>(2*1049866, vals);
  exp.AddKernel("single-threaded", spmv, vals_v);
  exp.AddKernel("omp-parallel", spmv_par, vals_v);
```

Not surprisingly, `AddKernel` function takes an identifier, a kernel function that follows the function definition `experiment::KernelFucntion`, and kernel specific parameters. Here as kernel specific parameters we pass dummy floats since the matrices we are using does not have any values.

### 5. Run the experiment.

Finally, run the experiment by calling the run function of the api.

```c++
  exp.Run(NUM_RUNS, true);
```

First parameter of `Run` take the number of times the experiment will be run. As parallel programmers we value the importance of averaging :).
If we wish to get the intermediate data generated, i.e. the permuted/reordered matrices, we can pass ``true`` as the second parameter of `Run`function.

### 6. Get results, runtimes, and auxiliary data.

After successfully running all the data with all the preprocessings and all the data generated with the kernels, we get the runtimes, auxiliary data, and the results using the `GetResults`, `GetAuxiliary`, and `GetRunTimes`functions.

```c++
  // check results
  cout << "Results: " << endl;
  auto res = exp.GetResults();
  for(auto r: res){
    cout << r.first << ": ";
    auto result = any_cast<float*>(r.second[0]);
    for(unsigned int t = 0; t < 50; t++){
      cout << result[t] << " ";
    }
    cout << endl;
  }

  cout << endl;

  // get auxiliary data created during the experiment
  auto auxiliary = exp.GetAuxiliary();
  cout << "Auxiliary Data: " << endl;
  for(const auto & a: auxiliary){
    cout << a.first << endl;
  }

  cout << endl;

  // display runtimes
  auto secs = exp.GetRunTimes();
  cout << "Runtimes: " << endl;
  for(const auto & s: secs){
    cout << s.first << ": ";
    for(auto sr: s.second){
      cout << sr << " ";
    }
    cout << endl;
  }
```

### 7. Compile & Execute the program.
Compile the code using `g++`. We assume SparseBase has already been installed in the compiled setting (as opposed to header-only installation).

While in the directory `tutorials/003_experiment/start_here`, execute the following commands:
```bash
g++ -std=c++17 tutorial_003.cc -lsparsebase -lgomp -fopenmp -std=c++17 -o experiment.out
./experiment.out ../../../examples/data/ash958.mtx ../../../examples/data/com-dblp.uedgelist 
```

You should see an output similar to the following:

```
[11/24/22 14:29:36] [WARNING] [sparsebase::format::COO<unsigned int, unsigned int, unsigned int>] COO arrays must be sorted. Sorting...
[11/24/22 14:29:36] [WARNING] [sparsebase::format::CSR<unsigned int, unsigned int, unsigned int>] CSR column array must be sorted. Sorting...
[11/24/22 14:29:42] [WARNING] [sparsebase::format::CSR<unsigned int, unsigned int, unsigned int>] CSR column array must be sorted. Sorting...
Results: 
-../../../examples/data/ash958.mtx,RCM,omp-parallel,0: 0.822065 0.0235149 0.626296 -0.0925408 -0.0329058 -0.26158 0.184576 -0.17292 0.527329 -0.632491 -0.371785 -0.218926 -0.24636 -0.401657 0.356059 0.556284 0.734213 0.538989 0.298418 0.380691 0.642242 -0.531986 -0.115331 -0.53883 0.268276 0.114501 -0.226813 0.44635 0.609854 0.412667 0.511964 0.489284 -0.211855 -0.660333 -0.706591 -0.195008 -0.145754 0.388777 -0.382022 -0.191022 -0.61304 0.570586 0.221566 -0.856259 -0.0557599 0.123087 -0.0439124 0.483856 -0.868368 0.0145038 
-../../../examples/data/ash958.mtx,RCM,single-threaded,0: 0.822065 0.0235149 0.626296 -0.0925408 -0.0329058 -0.26158 0.184576 -0.17292 0.527329 -0.632491 -0.371785 -0.218926 -0.24636 -0.401657 0.356059 0.556284 0.734213 0.538989 0.298418 0.380691 0.642242 -0.531986 -0.115331 -0.53883 0.268276 0.114501 -0.226813 0.44635 0.609854 0.412667 0.511964 0.489284 -0.211855 -0.660333 -0.706591 -0.195008 -0.145754 0.388777 -0.382022 -0.191022 -0.61304 0.570586 0.221566 -0.856259 -0.0557599 0.123087 -0.0439124 0.483856 -0.868368 0.0145038 
-../../../examples/data/ash958.mtx,original,omp-parallel,0: -0.773307 0.0264165 0.252196 0.489519 -0.105733 -0.294369 0.571206 0.695816 0.26244 -0.167607 -0.331523 -0.172566 -0.238891 0.0280457 0.315937 0.178022 0.762208 0.384923 0.0996124 0.583326 0.134168 -0.684496 -0.0302757 0.146844 0.110476 0.147265 -0.497262 -0.992332 0.470027 -0.30586 0.874546 0.128382 0.21321 -0.101168 0.759906 -0.108132 -0.148001 -0.331163 0.23896 -0.103248 0.221801 0.182518 -0.572703 0.633607 0.21792 0.220205 -0.167772 -0.103758 0.487433 -0.169161 
-../../../examples/data/ash958.mtx,original,single-threaded,0: -0.773307 0.0264165 0.252196 0.489519 -0.105733 -0.294369 0.571206 0.695816 0.26244 -0.167607 -0.331523 -0.172566 -0.238891 0.0280457 0.315937 0.178022 0.762208 0.384923 0.0996124 0.583326 0.134168 -0.684496 -0.0302757 0.146844 0.110476 0.147265 -0.497262 -0.992332 0.470027 -0.30586 0.874546 0.128382 0.21321 -0.101168 0.759906 -0.108132 -0.148001 -0.331163 0.23896 -0.103248 0.221801 0.182518 -0.572703 0.633607 0.21792 0.220205 -0.167772 -0.103758 0.487433 -0.169161 
-../../../examples/data/com-dblp.uedgelist,RCM,omp-parallel,0: 0.523699 0.0201702 -0.129043 -0.339734 0.00160382 0.897491 1.14721 -1.66401 -0.702446 -0.915523 -0.577063 -0.459128 0.392694 0.0368445 -0.0352359 0.620369 -0.88126 0.551372 -0.147413 -0.0913807 0.933404 -0.0427016 -0.764132 -0.0581159 0.182207 0.35385 0.00877577 0.237135 0.533137 0.0410516 -0.540313 1.14785 -0.633249 0.529273 -0.0955376 0.789746 0.353336 0.0572413 -0.257666 0.361567 0.13089 0.282823 -0.370485 -0.748098 0.503802 -0.505965 0.419774 0.182175 0.204978 0.152497 
-../../../examples/data/com-dblp.uedgelist,RCM,single-threaded,0: 0.523699 0.0201702 -0.129043 -0.339734 0.00160382 0.897491 1.14721 -1.66401 -0.702446 -0.915523 -0.577063 -0.459128 0.392694 0.0368445 -0.0352359 0.620369 -0.88126 0.551372 -0.147413 -0.0913807 0.933404 -0.0427016 -0.764132 -0.0581159 0.182207 0.35385 0.00877577 0.237135 0.533137 0.0410516 -0.540313 1.14785 -0.633249 0.529273 -0.0955376 0.789746 0.353336 0.0572413 -0.257666 0.361567 0.13089 0.282823 -0.370485 -0.748098 0.503802 -0.505965 0.419774 0.182175 0.204978 0.152497 
-../../../examples/data/com-dblp.uedgelist,original,omp-parallel,0: 1.20506 2.95578 -1.01716 -0.766262 -0.267771 0.06677 1.55544 0.220332 -0.571644 -0.168241 0.113864 -0.361985 -0.0664757 -0.126499 0.316209 0.801863 -0.464519 -0.653387 -0.0847772 -0.946206 0.796946 -0.168874 -0.617958 0.833564 -0.237255 0.130632 0.103078 0.238537 0.461887 0.130257 -0.183248 0.800289 0.310773 -0.693214 -1.27944 -0.0579325 0.851679 -0.354578 2.59094 0.128936 -0.000659117 -0.141935 0.365061 1.12694 0.282495 -0.0483253 -0.22286 1.98556 -1.08325 -0.0935952 
-../../../examples/data/com-dblp.uedgelist,original,single-threaded,0: 1.20506 2.95578 -1.01716 -0.766262 -0.267771 0.06677 1.55544 0.220332 -0.571644 -0.168241 0.113864 -0.361985 -0.0664757 -0.126499 0.316209 0.801863 -0.464519 -0.653387 -0.0847772 -0.946206 0.796946 -0.168874 -0.617958 0.833564 -0.237255 0.130632 0.103078 0.238537 0.461887 0.130257 -0.183248 0.800289 0.310773 -0.693214 -1.27944 -0.0579325 0.851679 -0.354578 2.59094 0.128936 -0.000659117 -0.141935 0.365061 1.12694 0.282495 -0.0483253 -0.22286 1.98556 -1.08325 -0.0935952 

Auxiliary Data: 
format,-../../../examples/data/ash958.mtx
format,-../../../examples/data/com-dblp.uedgelist
processed_format,-../../../examples/data/ash958.mtx,RCM
processed_format,-../../../examples/data/ash958.mtx,original
processed_format,-../../../examples/data/com-dblp.uedgelist,RCM
processed_format,-../../../examples/data/com-dblp.uedgelist,original

Runtimes: 
-../../../examples/data/ash958.mtx,RCM,omp-parallel,0: 0.00653002 
-../../../examples/data/ash958.mtx,RCM,single-threaded,0: 0.00593936 
-../../../examples/data/ash958.mtx,original,omp-parallel,0: 0.00605211 
-../../../examples/data/ash958.mtx,original,single-threaded,0: 0.00647483 
-../../../examples/data/com-dblp.uedgelist,RCM,omp-parallel,0: 0.0185962 
-../../../examples/data/com-dblp.uedgelist,RCM,single-threaded,0: 0.0150828 
-../../../examples/data/com-dblp.uedgelist,original,omp-parallel,0: 0.0171727 
-../../../examples/data/com-dblp.uedgelist,original,single-threaded,0: 0.0170771 
```

Check out the `examples/example_experiment` folder for an extended version, that uses GPUs, of the experiment we just created, ran and analyzed.
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
Last parameter of this function is the preprocessing specific parameters. Here the preprocessing algorithms we wish to use does not use any parameters.


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
Results: 
-../../../examples/data/ash958.mtx,RCM,omp-parallel,0: -0.171484 -0.260368 0.250003 -0.0173022 0.614765 -0.0825047 -0.681701 -0.182007 -0.347036 -0.657908 -0.650904 0.521015 -1.08387 0.162882 -0.178104 -0.123238 -0.110379 -0.248172 -0.506795 0.189518 0.444026 0.416654 0.391028 -0.00322095 0.200559 -0.11171 -0.303352 -0.0161204 0.412364 -0.397113 0.0224258 -0.0262049 0.726296 -0.654598 -0.21162 0.145143 0.390169 0.0969429 0.374616 0.216778 -0.98276 0.906358 -0.195527 0.633568 -0.468434 0.0589824 -0.918106 -0.932437 0.389362 0.131507 
-../../../examples/data/ash958.mtx,RCM,single-threaded,0: -0.171484 -0.260368 0.250003 -0.0173022 0.614765 -0.0825047 -0.681701 -0.182007 -0.347036 -0.657908 -0.650904 0.521015 -1.08387 0.162882 -0.178104 -0.123238 -0.110379 -0.248172 -0.506795 0.189518 0.444026 0.416654 0.391028 -0.00322095 0.200559 -0.11171 -0.303352 -0.0161204 0.412364 -0.397113 0.0224258 -0.0262049 0.726296 -0.654598 -0.21162 0.145143 0.390169 0.0969429 0.374616 0.216778 -0.98276 0.906358 -0.195527 0.633568 -0.468434 0.0589824 -0.918106 -0.932437 0.389362 0.131507 
-../../../examples/data/ash958.mtx,original,omp-parallel,0: 1.18348 0.532901 0.482204 1.33752 -0.116546 -0.486915 0.712344 -0.26227 -0.333573 -0.0256827 0.479494 0.244358 0.245474 0.0445811 -0.305284 -0.583675 0.275866 0.44404 -0.814038 -0.299801 0.447284 0.000714868 -0.0818595 -0.393439 0.366996 -0.748168 0.245914 0.783561 -0.680148 -0.20978 0.037306 0.118524 -0.688808 -0.0652988 -0.124971 -0.162297 0.0412879 0.00662421 -0.0179871 0.195053 0.33868 -1.13615 -1.09629 -0.226013 -0.605173 -0.237083 0.576704 -0.228609 0.398618 0.218906 
-../../../examples/data/ash958.mtx,original,single-threaded,0: 1.18348 0.532901 0.482204 1.33752 -0.116546 -0.486915 0.712344 -0.26227 -0.333573 -0.0256827 0.479494 0.244358 0.245474 0.0445811 -0.305284 -0.583675 0.275866 0.44404 -0.814038 -0.299801 0.447284 0.000714868 -0.0818595 -0.393439 0.366996 -0.748168 0.245914 0.783561 -0.680148 -0.20978 0.037306 0.118524 -0.688808 -0.0652988 -0.124971 -0.162297 0.0412879 0.00662421 -0.0179871 0.195053 0.33868 -1.13615 -1.09629 -0.226013 -0.605173 -0.237083 0.576704 -0.228609 0.398618 0.218906 
-../../../examples/data/com-dblp.uedgelist,RCM,omp-parallel,0: -0.0584508 -0.851059 -0.0859246 0.0426511 0.300712 0.14079 -0.186053 -0.542102 -0.0692798 -0.310758 -0.154896 -0.952702 -0.415188 0.0465407 -0.0774444 0.637579 0.363195 0.695686 -0.00187008 0.135546 -0.0348931 0.0245812 -0.36411 0.615983 -0.553137 -0.191394 0.233536 -0.232062 0.8511 0.550443 -0.0328045 -1.04605 -0.371555 -1.46733 -0.634119 0.0104862 0.215268 0.547263 -0.065832 -0.0499703 0.0235984 -0.529689 -0.702268 -0.969037 -0.375638 1.33601 -0.389561 0.598164 -0.515167 0.412709 
-../../../examples/data/com-dblp.uedgelist,RCM,single-threaded,0: -0.0584508 -0.851059 -0.0859246 0.0426511 0.300712 0.14079 -0.186053 -0.542102 -0.0692798 -0.310758 -0.154896 -0.952702 -0.415188 0.0465407 -0.0774444 0.637579 0.363195 0.695686 -0.00187008 0.135546 -0.0348931 0.0245812 -0.36411 0.615983 -0.553137 -0.191394 0.233536 -0.232062 0.8511 0.550443 -0.0328045 -1.04605 -0.371555 -1.46733 -0.634119 0.0104862 0.215268 0.547263 -0.065832 -0.0499703 0.0235984 -0.529689 -0.702268 -0.969037 -0.375638 1.33601 -0.389561 0.598164 -0.515167 0.412709 
-../../../examples/data/com-dblp.uedgelist,original,omp-parallel,0: -1.22997 -1.82336 -0.555305 -0.365009 -0.0263525 -0.00176735 0.833573 0.397792 0.286237 0.425817 -0.0381279 0.00680669 0.0113185 -2.06727 -0.59663 0.636445 0.242686 0.146184 -0.173515 1.19867 1.21207 0.167314 0.230791 0.609944 0.169749 -0.138795 -0.0116608 -0.0817233 0.0821181 1.00647 0.4433 0.36428 0.244311 2.73464 -0.624107 -1.89315 -0.622421 -1.06265 0.233888 0.261854 -0.20289 -1.16118 -1.17462 1.712 0.657661 -0.169337 -0.0217475 0.497298 0.426725 -0.971947 
-../../../examples/data/com-dblp.uedgelist,original,single-threaded,0: -1.22997 -1.82336 -0.555305 -0.365009 -0.0263525 -0.00176735 0.833573 0.397792 0.286237 0.425817 -0.0381279 0.00680669 0.0113185 -2.06727 -0.59663 0.636445 0.242686 0.146184 -0.173515 1.19867 1.21207 0.167314 0.230791 0.609944 0.169749 -0.138795 -0.0116608 -0.0817233 0.0821181 1.00647 0.4433 0.36428 0.244311 2.73464 -0.624107 -1.89315 -0.622421 -1.06265 0.233888 0.261854 -0.20289 -1.16118 -1.17462 1.712 0.657661 -0.169337 -0.0217475 0.497298 0.426725 -0.971947 

Auxiliary Data: 
format,-../../../examples/data/ash958.mtx
format,-../../../examples/data/com-dblp.uedgelist
processed_format,-../../../examples/data/ash958.mtx,RCM
processed_format,-../../../examples/data/ash958.mtx,original
processed_format,-../../../examples/data/com-dblp.uedgelist,RCM
processed_format,-../../../examples/data/com-dblp.uedgelist,original

Runtimes: 
-../../../examples/data/ash958.mtx,RCM,omp-parallel,0: 0.00632314 
-../../../examples/data/ash958.mtx,RCM,single-threaded,0: 0.00582036 
-../../../examples/data/ash958.mtx,original,omp-parallel,0: 0.00573753 
-../../../examples/data/ash958.mtx,original,single-threaded,0: 0.00571275 
-../../../examples/data/com-dblp.uedgelist,RCM,omp-parallel,0: 0.0183271 
-../../../examples/data/com-dblp.uedgelist,RCM,single-threaded,0: 0.0148009 
-../../../examples/data/com-dblp.uedgelist,original,omp-parallel,0: 0.0172842 
-../../../examples/data/com-dblp.uedgelist,original,single-threaded,0: 0.0173213
```

Check out the `examples/example_experiment` folder for an extended version, that uses GPUs, of the experiment we just created, ran and analyzed.
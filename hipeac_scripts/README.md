# Script Description
This folder contains the scripts used to generate the results reported in the study. All scripts are based on the original ones provided by the authors, with several modifications introduced to improve automation and correct various issues. All output results are stored in the `hipeac_results/` directory using `hipeac_` as prefix string for each result. Note that no change have been done on scripts under ```pattern/``` directory.


## Results Description
Results obtained by these scripts generate performance metrics in terms of execution time, as well as quality metrics that measure the number of hyperedge cuts cut(H) produced by each partitioner.

In fact, the primary objective of hypergraph partitioning is the maximization of partitioning quality. To compare the quality of different methods, it is used the normalized metric  
\[
(cut_A(H) - cut_B(H)) / |E|
\]
which measures, for each hypergraph \(H\), the relative difference in the number of hyperedges cut by method \(A\) versus method \(B\), normalized by the total number of hyperedges \(|E|\). This value indicates whether method \(A\) performs better (negative values), worse (positive values), or comparably to method \(B\).

Since hypergraph partitioning can be performed with different numbers of partitions, the study evaluates this metric for **k-way partitioning** with \(k = 2, 3, 4\).  
These configurations correspond to increasingly fine-grained partitions of the hypergraph, and the generated results therefore allow assessing how partitioning quality varies as the number of partitions increases.



## CPU-Based Work
The CPU hypergraph partitioners used as baselines are:
- `BiPart`
- `Mt-KaHyPar`
- `hMetis`

To reproduce the corresponding results, use the following scripts:
- `run-bipart.py` reproduces `BiPart` performance results 
- `run-bipart-quality.py`reproduces `BiPart` quality results
- `run-mtkahypar.py` reproduces `Mt-KaHyPar` performance and quality results (the switch is hard coded as a comment inside the script)
- `run-hmetis.py`reproduces `hMetis` performance and quality results in the same script

Note that each work has a different build process needed before running experiments. Everything could be build manually using `CMake` that is present in each directory, or standard compilation could be triggered simply using python òscripts.  

`hMetis` is present as a static 32 bit ELF file, therefore additional libraries could be needed in order to run its script. 

`Mt-KaHyPar` needs external `google-test` dependency that is automatically downloaded if using `build.sh` script provided. 

`BiPart` needs an external C++ library as dependency called Boost that is not  present in every linux environment and needs to be manually installed.


## GPU-Based Work
In the paper is presented a new work for GPU hypergraph partitioner `gHypart`. Since in the paper are presented different versions of the work, in particular a baseline first implementation `gHyPart-B`, a usable implementation based on the use of a decision tree in order to select the best parallelization strategy `gHyPart` and a implementation based on the best parallelization strategy `gHyPart-O` obtained as brute force of all possible combinations. This selection is used to improve performance. Moreover there is another selection based on policies in order to get a better quality on results.

### gHyPart
The following scripts are used to reproduce the results for the GPU-based hypergraph partitioner `gHyPart`:
- `run-gHypart.py` generates results for `gHyPart` and `gHyPart-B`
- `run_all_patterns.py` generates results for `gHyPart-O`

In particular `run-gHypart.py` make the user select which experiment to run using a minimalistic CLI, letting also the user choose between a standard policy or the best policy picked form authors' results file. 
# gHyPart_TACO
This is the source code repo for the original work "gHyPart: GPU-friendly End-to-End Hypergraph Partitioner" accepted by ACM TACO.

# Benchmarks
The benchmark set used in our work is available on [zenodo](https://zenodo.org/records/14822691) and can also be downloaded from [here](https://hkustgz-my.sharepoint.com/:u:/g/personal/zwu065_connect_hkust-gz_edu_cn/EUE2L_acRZdOsALyzbGZ018Bg-KzptQwsxrokg5ws9z3zA?e=CZ2CeA).

# Build CPU Baseline Works
`git clone --recurse-submodules https://github.com/zlwu92/gHyPart_TACO.git`

`cd cpu_works/BiPart/` and `cd cpu_works/Mt-KaHyPar-SDet/`

`mkdir build && cd build/`

`cmake .. -DCMAKE_BUILD_TYPE=RELEASE`

`make bipart-cpu -j` and `make MtKaHyPar -j`

# Build gHyPart
Go to gHyPart's root dir.

`mkdir build && cd build/ && cmake ..`

`make -j`

# Run experiments
Go to `scripts/` folder and use `python run-xxx.py` to run all experiments.

# Plot figures in paper
Go to `figure_plotting` folder and run all scripts, you can get the figures plotted in paper. 

---

# HiPEAC Student Challenge
This part of the ```README``` is an add on to the original work done for the HiPEAC student challenge 2025. The goal of this repository is to give an insight about what have been done in order to let experiments present in the paper "_gHyPart: GPU-friendly End-to-End Hypergraph Partitioner_" published on ACM journal on Transactions on Architecture and Code Optimization (TACO) be reproducable and verifiable.

## FIle Status
With respect to the original repository only ```.gitignore``` and ```CMakeList.txt``` have been changed since it was necessary to compile and run the code on different hardware and software toolchains. Every other change is present inside ```hipeac_results```, ```hipeac_figure_plotting``` and ```hipeac_scripts``` directories, containing respectively results obtained during the experimental phase of the review process, the scripts used in order to generate figures present in the report and scripts used in order to reproduce results.  

## Experiment Reproduction
In order to reproduce experiments, and therefore have an independent validation of the results, it is needed to download hypergraph dataset as previously stated by the authors, create a new directory called ```benchmark_set/``` and move all the dataset inside of it. In order to change dataset directory it is needed to change inside the ```hipeac_scripts/input_header.py``` 


## Contributors
The contributors for this repository and for the final work are three PhD students based in Tor Vergata University of Rome:
- Matteo Federico
- Simone Bauco
- Simone Staccone
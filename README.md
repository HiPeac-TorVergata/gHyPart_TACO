# gHyPart_TACO
This is the source code for the original work "gHyPart: GPU-friendly End-to-End Hypergraph Partitioner" accepted by ACM TACO.

# Build CPU Baseline Works
`git clone --recurse-submodules https://github.com/zlwu92/gHyPart_TACO.git`

`cd cpu_works/BiPart/` and `cd cpu_works/Mt-KaHyPar-SDet/`

`mkdir build && cd build/`

`cmake .. -DCMAKE_BUILD_TYPE=RELEASE`

`make bipart-cpu` and `make MtKaHyPar`

# Build gHyPart
Go to gHyPart's root dir.

`mkdir build && cd build/ && cmake ..`

`make`


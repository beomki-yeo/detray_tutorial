# Detray Tutorial

### Prerequistes
1. GCC >= 10.3.1
2. CUDA >= 11.5 and any NVIDIA GPU

### Clone the repository and Compile

```sh
git clone https://github.com/beomki-yeo/detray_tutorial.git
cd detray_tutorial
mkdir build
cd build/
cmake ../ -DCMAKE_BUILD_TYPE=Release
make
```

### Run propagator example

```sh
# CPU propagation
./bin/detray_tutorial_propagator_cpu

# CUDA propagation
./bin/detray_tutorial_propagator_cuda
```

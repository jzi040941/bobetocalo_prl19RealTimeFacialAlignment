# bobetocalo_prl19RealTimeFacialAlignment
RealTimeFacialAlignment using bobetocalo_prl19 model

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework
- Download model file
Download ModelFile from https://github.com/bobetocalo/bobetocalo_prl19
and place it under data/
- tensorflow
You have to build libtensorflow_cc.so with bazel

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── multitask
        └── RealtimeFacialAlignment
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir build
> cd build
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
Use the --database option to load the proper trained model.
```
> ./release/RealtimeFacialAlignment_test --database 300w_public
```

## Reference
```
Cascade of Encoder-Decoder CNNs with Learned Coordinates Regressor for Robust Facial Landmarks Detection
Roberto Valle, José M. Buenaposada and Luis Baumela.
Pattern Recognition Letters, PRL 2019.

https://github.com/bobetocalo/bobetocalo_prl19
```

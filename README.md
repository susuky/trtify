# trtify
Trtify: Simplifying PyTorch to TensorRT Model Conversion with Encryption. Effortlessly transform PyTorch models into TensorRT engines with added encryption capabilities for enhanced security


## Instalation

Normal instalation

```sh
git clone https://github.com/susuky/trtify.git
cd trtify
chmod +x setup.sh
./setup.sh
```

To cythonize

```sh
CYTHONIZE=1 ./setup.sh
```

To build wheel

```sh
BUILD_WHEEL=1 ./setup.sh
# pip install dist/trtify-*.whl
```

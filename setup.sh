
# Args:
    # BUILD_WHEEL: 1 or 0
    # CYTHONIZE: 1 or 0

# For example:
    # $ CYTHONIZE=1 BUILD_WHEEL=1 ./setup.sh
    # $ pip install dist/trtify-*.whl

rm dist -rf


if [ "$BUILD_WHEEL" = '1' ]; then
    python setup.py bdist_wheel 
    # pip install dist/trtify-*.whl
else
    pip install . 
fi



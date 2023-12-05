rm dist -rf

# python setup.py bdist_wheel
# pip install dist/trtify-*.whl

if [ -n "$CYTHONIZE" ]; then
    CYTHONIZE=$CYTHONIZE pip install .
else
    pip install .
fi
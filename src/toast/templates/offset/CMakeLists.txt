
# Install the python files

install(FILES
    __init__.py
    offset.py
    kernels.py
    kernels_numpy.py
    kernels_jax.py
    DESTINATION ${PYTHON_SITE}/toast/templates/offset
)

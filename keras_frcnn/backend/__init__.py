import keras

assert keras.backend.backend() in ['tensorflow'], 'Only tensorflow supported currently'

# Import backend functions based on your keras-backend
if keras.backend.backend() == 'tensorflow':
    from .tensorflow_backend import *

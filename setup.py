import setuptools

setuptools.setup(
    name='keras-pipeline',
    version='0.4',
    description='A collection of deep learning related models and tools written in keras and tensorflow',
    url='https://github.com/mingruimingrui/keras-pipeline',
    author='Wang Ming Rui',
    author_email='mingruimingrui@hotmail.com',
    packages=[
        'keras_pipeline',
        'keras_pipeline.backend',
        'keras_pipeline.utils',
        'keras_pipeline.layers',
        'keras_pipeline.models',
        'keras_pipeline.losses',
        'keras_pipeline.callbacks',
        'keras_pipeline.evaluation',
        'keras_pipeline.preprocessing',
        'keras_pipeline.generators'
    ]
)

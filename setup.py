from setuptools import setup


setup(
    name='stn',
    version='1.0.1',
    description='Spatial Transformer Networks.',
    long_description='Implementation of https://arxiv.org/abs/1506.02025',
    url='https://github.com/kevinzakka/spatial-transformer-network',
    author='Kevin Zakka',
    author_email='kevinarmandzakka@gmail.com',
    license='MIT',
    keywords='ai neural networks machine learning ml deep learning dl spatial transformer networks',
    packages=['stn'],
    install_requires=['numpy'],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"]
    }
)

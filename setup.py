from setuptools import setup, find_packages

setup(
    name='eranns',
    version='0.1',
    description='ERANNs: Efficient Residual Attention Neural Networks for Audio Classification',
    author='Matheus A. de Oliveira',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.18.0',
        'numpy>=1.26.4',
        'pandas>=2.2.3',
        'librosa>=0.11.0'
    ],
    python_requires='>=3.11',
)

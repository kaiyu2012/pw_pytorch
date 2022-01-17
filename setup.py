from setuptools import setup
from setuptools import find_packages

setup(name='SynapticFlowGNN',
      version='0.1',
      description='SynapticFlow Graph Convolutional Network in PyTorch',
      author='Po-wei Harn, Sai Deepthi',
      author_email='harnpowei@gmail.com',
      url='',
      download_url='',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy',
                        'matplotlib',
                        'sklearn',
                        'mlxtend',
                        'networkx',
                        'torchprofile',
                        'dgl-cu101==0.4.1'
                        ],
      package_data={'SynapticFlowGNN': ['README.md']},
      packages=find_packages())

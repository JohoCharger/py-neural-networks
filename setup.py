from setuptools import setup, find_packages

setup(
   name='py_neural_networks',
   version='1.0',
   description='A neural network implementation in Python',
   author='Jooans Lindroos',
   author_email='joonas.lindroos@outlook.com',
   packages=find_packages(),  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
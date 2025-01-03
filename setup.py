from setuptools import setup

setup(
   name='py-neural-networks',
   version='1.0',
   description='A neural network implementation in Python',
   author='Jooans Lindroos',
   author_email='joonas.lindroos@outlook.com',
   packages=['py-neural-networks'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
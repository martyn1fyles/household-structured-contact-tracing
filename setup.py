from setuptools import setup

setup(
   name='household-contact-tracing',
   version='0.5',
   description='Contact tracing model at the household level.',
   license="GNU",
   #long_description=long_description,
   #author='Man Foo',
   #author_email='foomail@foo.com',
   #url="http://www.foopackage.com/",
   #packages=['household-contact-tracing'],  #same as name
   install_requires=['networkx', 'numpy', 'scipy', 'matplotlib', 'pytest', 'pydot'], #external packages as dependencies
   py_modules = ['household_contact_tracing']
)

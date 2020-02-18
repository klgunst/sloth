Some notes about pyT3NS
========================

A sloth is slowly moving through a tree.

'Purely' pythonic code for T3NS calculations or other three-legged
calculations.

Network object:
* Has the different tensors as nodes of some connected graph. vacuum taret and
  physical legs are also nodes of this connected graph. Swapping tensors as now
  as easy as swapping nodes.

Tensor class (Three-legged):
* dict with tuple of symmetries as keys and with numpy arrays and dimensions as
  values.

Let's try first a class that reads the corresponding HDF5 File of a previous
calc into python

can swap the trees around and write again the new HDF5. That would be nice.

This whole mumbo jumbo van ordering of indexes en qnumbers is different I
should take into account?

Of met loose en internals werken en coupling ook?

# Variational-MPS-with-Julia

Under construction!

Read first:

**functions_graph_variational_ground_state_MPS.pdf**

and

**Description of functions in variational_ground_state_MPS.pdf**

to understand which function calls which and how the variables created are passed around the program in the **variational_MPS_algorithm.jl** file. These two pdf documents should be read at the same time.

Current vesion can: perform the variational ground state MPS algorithm and get the ground state energy and ground state MPS for a given input Hamiltonian, using the file **variational_MPS_algorithm.jl**. It can also measure the total spin along a given x,y or z axis.

The Benchmark Timer shown in the picture of the file **Benchmark Timer for variational_ground_state_MPS function.JPG** has input N = 4, D = 2, d = 2 for number of lattice sites, bond dimension and physical dimension respectively, computed on my humble machine Dell XPS 13.

This repository is for my project at the Cyprus Institute with Dr Stefan Kuhn. The aim is to build MPS Julia code for the variational MPS algorithm that usually
gets the ground state of the input Hamiltonian. Then we will apply it to the 1D transverse/longitudinal Ising model and finally investigate confinement in the presence
of a topological theta term in the Schwinger model.

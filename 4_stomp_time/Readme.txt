The directory 4_stom_time contains all files that solve the case
of fluctuations in incoming time for the spectrum model

The files contain:

generate_data*
	Generate data for different kinds of incoming light fluctuations

jobarray_rand*
	Bash scripts that call their respective generate_data file on the clusters

richness_computation
	Generates random communities and computes their richness under fluctuating incoming light

ode_solving
	Adam-Bashforth method for solving multiple odes at the same time parallelized

plot_figure*
	Plot the corresponding figures

connection_data
	A executable from the command line to connect data generated on the cluster

I_in_functions
	Generate different functions for incoming light intensity




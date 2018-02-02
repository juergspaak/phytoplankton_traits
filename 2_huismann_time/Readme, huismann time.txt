This folder contains functions and files to compute the boundary growth rates of species in constant incoming light intensity.

communities_*:
	contains functions for the generation of communities and some accompaning functions. 
	These functions are used by their respective r_i* files

r_i_*:
	Compute the boundary growth rates of the species using respectively numerical and analytical methods.
	_continuous uses continuous change of incoming light (periodic functions)
	_step uses step wise changes of incoming light, using randomized light over time
	_step files might need revision

evolutionary_stability_of_coexistence:
	Finds communities that do coexist and randomly changes their parameters slightly (evolution)
	After doing so checks again whether those slightly changed communities do coexist.
	
real_I_out
	Computes the boundary growth rate with numerical methods. Does not assume that I_out is equal to 0

real_background
	Computes the boundary growth rate with numerical methods. Does not assume that I_out is 0 nor that background is negligible
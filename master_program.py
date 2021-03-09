"""
Run this file to simulate the communities and create the images
This will take about 30-40 minutes

Important, the simulations are based on randomness, you will therefore not
obtain the identical dataset as used in the paper

Additionally, the code only creates about 1000 communities for time reasons
The obtained graphs may therefore differ from the ones in the paper.
To increase the number of communities simulated increase the running time
in sim_data_photoprotection on line 229.
E.g. run_time_in_seconds = 3600 will result in an hour of computation time
Optimally, you run the program sim_data_photoprotection multiple times in
parallel with changing the save parameter as well
"""

# simulate the phytoplankton communities, takes about 30 minutes
# running length can be adjusted, see above
import sim_data_photoprotection

# create all plots
# takes some few minutes
import plot_figure_1_absorption_spectra
import plot_figure_2_richness
import plot_figure_3_EF
import plot_figure_4_traits_NFD_barplot

import plot_figure_S1_I_out
import plot_figure_S2_complementarity
import plot_figure_S3_NFD_individually
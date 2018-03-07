#!/bin/bash


#############################################
#                                           #
# This is a script to create all the plots  #
# from the dart_board paper, a version of   #
# which is included in the directory        #
# dart_board/paper/docs                     #
#                                           #
# To use, the relevant simulations must     #
# have first been run, and the output       #
# files must exist in dart_board/paper/data #
# Once these files exist, simply uncomment  #
# the relevant lines below to create your   #
# figures.                                  #
#                                           #
# Jeff J. Andrews                           #
# andrews@physics.uoc.gr                    #
# University of Crete                       #
#                                           #
#############################################

###### Figure 2 ######

# python3 LMC_SFH_plot.py


###### Figure 4 ######

# python3 corner_plot.py mock_1_low


###### Figure 5 ######

# python3 corner_plot.py mock_2_low


###### Figure 6 ######

# python3 corner_plot.py mock_3_PT_low


###### Figure 7 ######

# python3 corner_plot.py HMXB


###### Figure 8 ######

# python3 derived_plot.py HMXB


###### Figure 9 ######

# python3 corner_plot.py LMC_HMXB


###### Figure 10 ######

# python3 position_plot.py LMC_HMXB


###### Figure 11 ######

# python3 corner_plot_2dist.py J0513


###### Figures 12 & 17 ######

# python3 J0513_plots_2.py


###### Figures 13 & 18 ######

# python3 J0513_plots_2.py 200 high


###### Figure 14 ######

# python3 corner_plot_2dist.py J0513_flatsfh


###### Figure 15 ######

# python3 chains_plot.py HMXB 20000 40000 100 7
# convert -density 200 ../figures/HMXB_chains.pdf ../figures/HMXB_chains.jpg


###### Figure 16 ######

# python3 acor_plot_2panel.py HMXB J0513_low


###### Figure 19 ######

# python3 HMXB_plots_2.py


###### Figures 20 & 21 ######

# These figures are formed within the python notebook Channel ratios.ipynb
# found in the directory dart_board/paper/notebook/
# See this notebook for more details.




###### Other potentially useful plots #####

# python3 chains_plot.py J0513_low 0 500000 1000 6
# convert -density 200 ../figures/J0513_low_chains.pdf ../figures/J0513_low_chains.jpg

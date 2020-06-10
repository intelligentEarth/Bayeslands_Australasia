import os
import sys
import random
import operator
import math 
import copy
import fnmatch
import shutil
import time
import numpy as np
from badlands.model import Model as badlandsModel
import pickle

def ProblemSetup():
    
    #Randomize seed value. Run with new seed every time.
    random.seed(time.time()) 

    directory = 'Examples/australia/'
    xml_filepath = directory + 'AUSB001.xml'
    init_topo_expert = np.loadtxt(directory + 'data/init_topo_fused.txt')  # no expert knowledge as simulated init topo    
    init_topo_estimate = []  #init_topo_estimate = #np.loadtxt(directory + 'init_expertknowlegeprocess/init_estimated.txt')

    simtime = -1.49e08
    resolu_factor = 1 

    init_elev = np.loadtxt(directory+ 'data/initial_elev.txt')

    true_elev = np.loadtxt(directory +'data/final_elev_filtered_ocean.txt')
    true_erdp = np.loadtxt(directory +'data/final_erdp.txt')

    true_elev_pts = np.loadtxt(directory +'data/elev_pts_updated.txt')
    true_erdp_pts = np.loadtxt(directory +'data/final_erdp_pts_.txt')

    sea_level = np.loadtxt(directory+ 'AUSB001/Sea_level/Muller2018_M6.csv')  

    erdp_coords = np.loadtxt(directory +"data/erdp_coords.txt", )   
    print('No. of points for erosion deposition likl (erdp_coords): ', erdp_coords.shape)
    erdp_coords = np.array(erdp_coords, dtype = 'int')

    elev_coords = np.loadtxt(directory +"data/coord_final_elev.txt", )  
    print('No. of points for elevation likl (elev_coords)', elev_coords.shape)        
    elev_coords = np.array(elev_coords, dtype = 'int')

    result_summary = '/results.txt'

    #true_parameter_vec = np.loadtxt(directory + 'data/true_values.txt')
    sediment_likl = True

    rain_min = 0.3
    rain_max = 1.8

    # assume 1 regions and 4 time scales
    rain_regiongrid = 1  # how many regions in grid format 
    rain_timescale = 4  # to show climate change 
    rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
    rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 

    #--------------------------------------------------------
    parameters_fixed = False

    if parameters_fixed:
        minlimits_others = [1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01, 0]  # used for Bayeslands environmental params  (stage 2) 
        maxlimits_others = [1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01, 80] #   429.560216846004 rmse_elev   2921.1315327463903 rmse_erdep

    else:
        minlimits_others = [5.e-7, 0 , 0 , 0  ,  0  , 0  ,  0 , 0 , 20000, 4 , 0 , 0  ]  # used for Bayeslands environmental params  (stage 2) 
        maxlimits_others = [5.e-6, 1 , 2 , 0.5, 0.05, 0.05, 1 , 20 , 30000, 20 , 0.2 , 80]

    #variables[:15] = [1.16, 0.9, 1.092, 1.0, 1.e-6, 0.5, 1.0, 0.005, 0.001, 0.001, 0.5, 5, 24000, 5, 0.01]

    #----------------------------------------InitTOPO

    epsilon = 0.5 

    init_topo_inference = False

    if init_topo_inference:
        init_topo_gridlen = 7  # should be of same format as @   init_topo_expert
        init_topo_gridwidth = 7

        len_grid = int(true_elev.shape[0]/init_topo_gridlen)  # take care of left over
        wid_grid = int(true_elev.shape[1]/init_topo_gridwidth)   # take care of left over

    else:
        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore

    # print(len_grid, wid_grid, true_elev.shape[0], true_elev.shape[1] ,'  sub_gridlen, sub_gridwidth   ------------ ********')

    inittopo_minlimits = np.repeat(-1 , 51)
    inittopo_maxlimits = np.repeat(1 , 51)

    sea_level_inference = True

    sealevel_max = [0.2,0.2,0.3,0.3,0.40,0.40,0.45,0.45,0.5,0.5] 
    sealevel_min = [-0.2,-0.2,-0.3,-0.3,-0.40,-0.40,-0.45,-0.45,-0.5,-0.5]

    #--------------------------------------------------------
    print('Parameters Fixed    : %r' %(parameters_fixed))
    print('Init Topo Inference : %r' %(init_topo_inference))
    print('Sea level Inference : %r' %(sea_level_inference))

    # Appending all parameter limits together into one vector
    minlimits_vec = np.append( rain_minlimits,minlimits_others )#,inittopo_minlimits)
    maxlimits_vec = np.append( rain_maxlimits,maxlimits_others )#,inittopo_maxlimits)

    minlimits_vec = np.append( minlimits_vec, sealevel_min)#,inittopo_minlimits)
    maxlimits_vec = np.append( maxlimits_vec, sealevel_max)#,inittopo_maxlimits)

    minlimits_vec = np.append( minlimits_vec, inittopo_minlimits)
    maxlimits_vec = np.append( maxlimits_vec, inittopo_maxlimits)

    print('Parameter distribution: ')
    print('Rain  : ', rain_maxlimits.shape[0], '  Others (erod,m,n ..) :', len(maxlimits_others),
    '  Sea level: ', len(sealevel_max), '  Init Topo: ', inittopo_maxlimits.shape[0] )
    vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
    # true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
    print('Minimum and Maximum limit vectors Shape : ', minlimits_vec.shape ,maxlimits_vec.shape)
    print('Parameter Vector Shape', vec_parameters.shape) 

    stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

    print('Step Size Ratio: %s' %stepsize_ratio)
    print('\n++++++++++++++++++++++++   Problem Setup Complete   ++++++++++++++++++++++++++++++++++++++')

    return (directory, xml_filepath, simtime, resolu_factor, sea_level, init_elev ,true_elev, true_erdp,
    true_erdp_pts, true_elev_pts,  result_summary, init_topo_expert, len_grid, wid_grid, sediment_likl, rain_min, rain_max, rain_regiongrid, minlimits_others,
    maxlimits_others, stepsize_ratio, erdp_coords, elev_coords, init_topo_estimate, vec_parameters,minlimits_vec, maxlimits_vec)
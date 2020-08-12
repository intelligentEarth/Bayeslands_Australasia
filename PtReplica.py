#Main Contributers:   Rohitash Chandra and Danial Azam  Email: c.rohitash@gmail.com 

# Bayeslands II: Parallel tempering for multi-core systems - Badlands

from __future__ import print_function, division
import os
import sys
import random
import time
import math 
import copy
import json
import collections
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import multiprocessing
import itertools
import chart_studio
import chart_studio.plotly as py
import pandas
import argparse
import subprocess
import pickle
import pandas as pd
import scipy.ndimage as ndimage
from plotly.graph_objs import *
from pylab import rcParams
from copy import deepcopy 
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats
from scipy import special

import badlands 
from badlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.ndimage import filters 
from scipy.ndimage import gaussian_filter
from ProblemSetup import ProblemSetup

from GenUplift import edit_Uplift, process_Uplift
from GenInitTopo import process_inittopoGMT, edit_DBF

class PtReplica(multiprocessing.Process):
    
    def __init__(self, ID, num_param, vec_parameters, sea_level, ocean_t,  inittopo_expertknow, rain_region, rain_time, len_grid, wid_grid, minlimits_vec, maxlimits_vec, stepratio_vec,   check_likelihood_sed ,  swap_interval, sim_interval, simtime, samples, init_elev, real_elev,  real_erodep_pts, real_elev_pts, erodep_coords,elev_coords, filename, xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in, inittopo_estimated, covariance, Bayes_inittopoknowledge, uplift, inittopo):

        self.ID = ID
        multiprocessing.Process.__init__(self)
        self.processID = tempr      
        self.parameter_queue = parameter_queue
        self.event = event
        self.signal_main = main_proc
        self.temperature = tempr
        self.swap_interval = swap_interval
        self.folder = filename
        self.input = xmlinput  
        self.simtime = simtime
        self.samples = samples
        self.run_nb = run_nb 
        self.num_param =  num_param
        self.font = 9
        self.width = 1 
        self.vec_parameters = np.asarray(vec_parameters)
        self.minlimits_vec = np.asarray(minlimits_vec)
        self.maxlimits_vec = np.asarray(maxlimits_vec)
        self.stepratio_vec = np.asarray(stepratio_vec)
        self.check_likelihood_sed =  check_likelihood_sed
        self.real_erodep_pts = real_erodep_pts
        self.real_elev_pts = real_elev_pts
        self.elev_coords = elev_coords
        self.erodep_coords = erodep_coords
        self.ocean_t = ocean_t
        self.init_elev = init_elev
        self.real_elev = real_elev
        self.runninghisto = True  
        self.burn_in = burn_in
        self.sim_interval = sim_interval
        self.sedscalingfactor = 1 # this is to ensure that the sediment likelihood is given more emphasis as it considers fewer points (dozens of points) when compared to elev liklihood (thousands of points)
        self.adapttemp =  self.temperature
        self.rain_region = rain_region 
        self.rain_time = rain_time 
        self.len_grid = len_grid 
        self.wid_grid  = wid_grid# for initial topo grid size 
        self.inittopo_expertknow =  inittopo_expertknow 
        self.inittopo_estimated = inittopo_estimated
        self.adapt_cov = 50
        self.cholesky = [] 
        self.cov_init = False
        self.use_cov = covariance
        self.cov_counter = 0
        self.repeated_proposal = False
        self.sealevel_data = sea_level
        self.Bayes_inittopoknowledge = Bayes_inittopoknowledge
        self.uplift = uplift
        self.inittopo = inittopo

    def init_show(self, zData, fname, replica_id): 
 
        fig = plt.figure()
        im = plt.imshow(zData, cmap='hot', interpolation='nearest')
        plt.colorbar(im)
        plt.savefig(self.folder + fname+ str(int(replica_id))+'.png')
        plt.close()

    def computeCovariance(self, i, pos_v):
        cov_mat = np.cov(pos_v[:i,].T) 

        cov_noise_old = (self.stepratio_vec * self.stepratio_vec)*np.identity(cov_mat.shape[0], dtype = float)
        cov_noise = self.stepsize_vec*np.identity(cov_mat.shape[0], dtype = float)
        covariance = np.add(cov_mat, cov_noise)        
        L = np.linalg.cholesky(covariance)
        self.cholesky = L
        self.cov_init = True
        # self.cov_counter += 1 

    def process_sealevel(self, coeff):

        y = self.sealevel_data[:,1].copy()
        timeframes = self.sealevel_data[:,0]

        first = y[0:50] # sea leavel for 0 - 49 Ma to be untouched 
        second = y[50:] # this will be changed by sea level coeefecients proposed by MCMC 

        second_mat = np.reshape(second, (10, 10)) 

        updated_mat = second_mat

        # print(coeff, ' coeff -----------------')

        for l in range(0,second_mat.shape[0]):
            for w in range(0,second_mat.shape[1]): 
                updated_mat[l][w] =  (second_mat[l][w] * coeff[l]) +  second_mat[l][w]


        #print(updated_mat, '   updated ----------------------------- ')

        reformed_sl = updated_mat.flatten()
        combined_sl = np.concatenate([first, reformed_sl]) 
        #print(proposed_sealevel, proposed_sealevel.shape,  '  proposed_sealevel  proposed_sealevel.shape            ----------------------------- ')
        #https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
        yhat = self.smooth(combined_sl, 10)

        fig, ax =  plt.subplots()  
        fnameplot = self.folder +  '/recons_initialtopo/'+str(int(self.temperature*10))+'_sealevel_data.png' 
        ax.plot(timeframes, self.sealevel_data[:,1], 'k--', label='original')
        ax.plot(timeframes, combined_sl, label='perturbed')
        ax.plot(timeframes, yhat, label='smoothened')
        ax.legend()
        plt.savefig(fnameplot)
        plt.close()    

        proposed_sealevel = np.vstack([timeframes, yhat])

        return proposed_sealevel

    def smooth(self, y, box_pts):
        #https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
        #print(y.shape, y, ' ++ y ')
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def interpolateArray(self, coords=None, z=None, dz=None):
        """
        Interpolate the irregular spaced dataset from badlands on a regular grid.
        """
        x, y = np.hsplit(coords, 2)
        dx = (x[1]-x[0])[0]
        nx = int((x.max() - x.min())/dx+1 - 2)
        ny = int((y.max() - y.min())/dx+1 - 2)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))

        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        if len(z[indices].shape) == 3:
            z_vals = z[indices][:,:,0]
            dz_vals = dz[indices][:,:,0]
        else:
            z_vals = z[indices]
            dz_vals = dz[indices]

        zi = np.average(z_vals,weights=(1./distances), axis=1)
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)
        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0]]
            dzi[onIDs] = dz[indices[onIDs,0]]
        zreg = np.reshape(zi,(ny,nx))
        dzreg = np.reshape(dzi,(ny,nx))
        return zreg,dzreg

    def vector_dump(self, pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec, pred_elev_pts_vec):
        x =self.folder + "/realtime_data/" +str(self.ID)  + "/pred_elev_vec.pkl" 
        np.save(self.folder + "/realtime_data/" +str(self.ID)  + "/real_erodep_pts_vec.npy", self.real_elev_pts) # save

        f = open(x,"wb")
        pickle.dump(pred_elev_vec,f)
        f.close()

        g = open(self.folder + "/realtime_data/" +str(self.ID)  + "/pred_erodep_pts_vec.pkl", "wb") # save
        pickle.dump(pred_erodep_pts_vec,g)
        g.close()

        h = open(self.folder + "/realtime_data/" +str(self.ID)  + "/pred_elev_pts_vec.pkl","wb") # save
        pickle.dump(pred_elev_pts_vec,h)
        h.close()

        z = open(self.folder + "/realtime_data/" +str(self.ID)  + "/pred_erodep_vec.pkl","wb")
        pickle.dump(pred_erodep_vec,z)
        z.close()

        return

    def run_badlands(self, input_vector):
        #Runs a badlands model with the specified inputs

        # print(self.real_elev.shape, ' real evel sh')
        # print(self.real_elev_pts.shape, ' real evel pt sh')
 
        rain_regiontime = self.rain_region * self.rain_time # number of parameters for rain based on  region and time 

        #Create a badlands model instance
        model = badlandsModel()

        if self.uplift == 1:
            xml_id = int(self.ID)
        else:
            xml_id = 0

        print(xml_id, input_vector[11], ' xml_id  input_vector[11]')

        xmlinput = self.input[xml_id]

        #----------------------------------------------------------------
        # Load the XmL input file
        model.load_xml(str(self.run_nb), xmlinput, verbose =False, muted = True)

        num_sealevel_coef = 10

        if self.inittopo == 1:

            geoparam  = num_sealevel_coef + rain_regiontime+13  # note 10 parameter space is for erod, c-marine etc etc, some extra space ( taking out time dependent rainfall)
            inittopo_vec = input_vector[geoparam:]

            filename=xmlinput.split("/")
            problem_folder=filename[0]+"/"+filename[1]+"/"

            #Use the coordinates from the original dem file
            #Update the initial topography 
            xi=int(np.shape(model.recGrid.rectX)[0]/model.recGrid.nx)
            yi=int(np.shape(model.recGrid.rectY)[0]/model.recGrid.ny)
            #And put the demfile on a grid we can manipulate easily
            elev=np.reshape(model.recGrid.rectZ,(xi,yi)) 
            
            # dummy_file = pd.read_csv('AUS/upliftvariables.txt')
            # edit_Uplift(self.ID, dummy_file)
            # process_Uplift(self.ID)
 
            self.process_inittopoGMT(inittopo_vec)  
            init_filename='init_topo_polygon/Paleotopo_P100_50km_prec2_'+ str(int(self.ID)) +'.csv' 
            # upl_filename = 'AUS/%s/AUS001.xml'%(self.ID)
            #elev_framex = np.vstack((model.recGrid.rectX,model.recGrid.rectY,inittopo_estimate.flatten()))
            #np.savetxt(filename, elev_framex.T, fmt='%1.2f' ) 
            
            model.input.demfile=init_filename

            model._build_mesh(model.input.demfile, verbose=False) 

        model.force.rainVal[:] = input_vector[0:rain_regiontime] 

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[rain_regiontime]  
        model.flow.erodibility.fill(input_vector[rain_regiontime])

        # Adjust m and n values
        model.input.SPLm = input_vector[rain_regiontime+1]  
        model.input.SPLn = input_vector[rain_regiontime+2] 

        #Check if it is the etopo extended problem
        #if problem == 4 or problem == 3:  # will work for more parameters
        model.input.CDm = input_vector[rain_regiontime+3] # submarine diffusion
        model.input.CDa = input_vector[rain_regiontime+4] # aerial diffusion

        model.slp_cr = input_vector[rain_regiontime+5]
        model.perc_dep = input_vector[rain_regiontime+6]
        model.input.criver = input_vector[rain_regiontime+7]
        model.input.elasticH = input_vector[rain_regiontime+8]
        model.input.diffnb = input_vector[rain_regiontime+9]
        model.input.diffprop = input_vector[rain_regiontime+10]

        sealevel_coeff = input_vector[rain_regiontime+12 : rain_regiontime+12+ num_sealevel_coef] 

        # print(sealevel_coeff, ' sealevel_coeff ')

        model.input.curve = self.process_sealevel(sealevel_coeff)
 
        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()
        elev_pts_vec = collections.OrderedDict()

        model.run_to_time(-1.489999e08, muted = True)
        elev_, erodep_ = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff) 

        self.init_show(elev_, '/pred_plots/GMTinit_', self.ID )   

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]
            model.run_to_time(self.simtime, muted = True)

            elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros(self.erodep_coords.shape[0])
            elev_pts = np.zeros(self.elev_coords.shape[0])

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]

            for count, val in enumerate(self.elev_coords):
                elev_pts[count] = elev[val[0], val[1]]
 
            print('Sim time: ', self.simtime  , "   Temperature: ", self.temperature)
            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts
            elev_pts_vec[self.simtime] = elev_pts
 
        return elev_vec, erodep_vec, erodep_pts_vec, elev_pts_vec

    def likelihood_func(self,input_vector): 

        pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec, pred_elev_pts_vec = self.run_badlands(input_vector)
        
        self.vector_dump(pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec, pred_elev_pts_vec)

        likelihood_elev_ocean = 0
        rmse_ocean = np.zeros(self.sim_interval.size)
        pred_topo_present_day = pred_elev_vec[self.simtime] 
        pred_elev_vec_ = copy.deepcopy(pred_elev_vec) #pred_elev_vec.copy()
        #for i in range(6,self.sim_interval.size): #this caters for first half

        for i in range(len(self.sim_interval)): # to cater for beyond 60 Ma
            p_elev_ocean = pred_elev_vec_[self.sim_interval[i]] 
            pred_elev = pred_elev_vec[self.sim_interval[i]]
            pred_erdp = pred_erodep_vec[self.sim_interval[i]]
            r_elev_ocean = self.ocean_t[i,:,:]

            p_elev_ocean[p_elev_ocean>0] = 0
            p_elev_ocean[p_elev_ocean<0] = 1 

            matches = np.count_nonzero(p_elev_ocean==r_elev_ocean)
            non_matches = p_elev_ocean.size -matches

            print('\n sim_interval[i] ', self.sim_interval[i], ' matches : ', matches ,'  non matches : ', non_matches, 'percentage non match', (non_matches/p_elev_ocean.size)*100)
            fig = plt.figure()
            im = plt.imshow(p_elev_ocean, cmap='hot', interpolation='nearest')
            plt.colorbar(im)
            plt.savefig(self.folder +'/realtime_plots/' + str(self.ID)+'/' + str(self.sim_interval[i]) +'r_elev_ocean.png')
            plt.close()

            fig = plt.figure()
            im = plt.imshow(r_elev_ocean, cmap='hot', interpolation='nearest') 
            plt.colorbar(im)
            plt.savefig(self.folder +'/realtime_plots/' + str(self.ID)+'/' + str(self.sim_interval[i]) +'r_elev_ocean.png')
            plt.close()

            fig = plt.figure()
            im = plt.imshow(pred_erdp, cmap='hot', interpolation='nearest')
            plt.colorbar(im)
            plt.savefig(self.folder +'/realtime_plots/' + str(self.ID)+'/' + str(self.sim_interval[i]) +'erodep.png')
            plt.close()
            
            fig = plt.figure()
            im = plt.imshow(pred_elev, cmap='hot', interpolation='nearest')
            plt.colorbar(im)
            plt.savefig(self.folder +'/realtime_plots/' + str(self.ID) + '/'+str(self.sim_interval[i]) +'elev.png')
            plt.close()
            print('p_elev_ocean', p_elev_ocean.shape)
            print('r_elev_ocean', r_elev_ocean.shape)
            print('self.real_elev', self.real_elev.size)
            tausq_ocean = np.sum(np.square(p_elev_ocean - r_elev_ocean))/self.real_elev.size  
            rmse_ocean[i] = tausq_ocean
            likelihood_elev_ocean  += np.sum(-0.5 * np.log(2 * math.pi * tausq_ocean) - 0.5 * np.square(p_elev_ocean - r_elev_ocean) /  tausq_ocean )

        '''
        fig = plt.figure()
        plt.imshow(self.real_erodep_pts, cmap='hot', interpolation='nearest')
        plt.savefig(self.folder +'/realtime_plots/' + str(self.ID) + '/'+ str(self.sim_interval[i]) +'real_erodep.png')
        plt.close()
        '''
        fig = plt.figure()
        im = plt.imshow(self.real_elev, cmap='hot', interpolation='nearest')
        plt.colorbar(im)
        plt.savefig(self.folder +'/realtime_plots/' + str(self.ID)  +'/real_elev.png')
        plt.close()

        tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size 
        ## CHeck this
        likelihood_elev  = np.sum(-0.5 * np.log(2 * math.pi * tausq ) - 0.5 * np.square(pred_elev_vec[self.simtime] - self.real_elev) / tausq )  

        mean_erdep = np.mean(self.real_erodep_pts)
        erdep_predicted = pred_erodep_pts_vec[self.simtime] 

        ## CHECK THIS
        erdep_predicted[erdep_predicted < 0] = 0 

        tau_erodep  =  np.sum(np.square(erdep_predicted - self.real_erodep_pts))/ self.real_erodep_pts.shape[0]

        pred_elev = pred_elev_vec[self.simtime] 

        real_elev_filtered = np.where((self.real_elev>0) & (self.real_elev<1000), self.real_elev, 0)  
        pred_elev_filtered = np.where((pred_elev>0) & (pred_elev<1000), pred_elev, 0)

        diff = pred_elev_filtered  - real_elev_filtered
        count = np.count_nonzero(diff) 

        tau_elev =  np.sum(np.square(diff)) / count

        likelihood_elev  = np.sum(-0.5 * np.log(2 * math.pi * tau_elev ) - 0.5 * np.square(diff) / tau_elev )
        likelihood_erodep  = np.sum(-0.5 * np.log(2 * math.pi * tau_erodep ) - 0.5 * np.square(erdep_predicted - self.real_erodep_pts) / tau_erodep ) # only considers point or core of erodep    

        print('likelihood_elev', likelihood_elev, 'likelihood_erodep', likelihood_erodep,'likelihood_elev_ocean', likelihood_elev_ocean/4 )
        likelihood_ = likelihood_elev + likelihood_erodep + (likelihood_elev_ocean/4)
        #rmse_ocean = 0
        rmse_elev = np.sqrt(tau_elev)
        rmse_elev_ocean = np.average(rmse_ocean)
        rmse_erodep = np.sqrt(tau_erodep) 
        rmse_elev_pts = np.sqrt(tau_elev) 

        likelihood = likelihood_*(1.0/self.adapttemp)

        pred_topo_present_day = pred_elev_vec[self.simtime]
        #self.plot3d_plotly(pred_topo_presentday, '/pred_plots/pred_badlands_', self.temperature *10)    # Problem exists here XXXXXXX

        print('LIKELIHOOD :--: Elev: ',likelihood_elev, '\tErdp: ', likelihood_erodep, '\tOcean:',likelihood_elev_ocean,'\tTotal: ', likelihood_, likelihood)
        print('RMSE :--: Elev ', rmse_elev, 'Erdp', rmse_erodep, 'Ocean', rmse_elev_ocean)
        np.savetxt(self.folder +'/realtime_plots/' +str(self.ID)  + '/rmse_stats.txt', [rmse_elev, rmse_erodep, rmse_elev_ocean])
        np.savetxt(self.folder +'/realtime_plots/'+ str(self.ID)  + '/rmse_ocean_time.txt',rmse_ocean)
        return [likelihood, pred_elev_vec, pred_erodep_pts_vec, likelihood, rmse_elev_pts, rmse_erodep, rmse_ocean, rmse_elev_ocean ]

    def run(self):
        #This is a chain that is distributed to many cores. AKA a 'Replica' in Parallel Tempering

        self.init_show(self.real_elev, '/recons_initialtopo/real_evel', 1)
        self.init_show(self.init_elev, '/recons_initialtopo/expert_inittopo', 1)

        # fnameplot = self.folder +  '/recons_initialtopo/'+'scatter_erodep_.png' 
        # plt.scatter(self.erodep_coords[:,0], self.erodep_coords[:,1], s=2, c = 'b')
        # plt.scatter(self.elev_coords[:,0], self.elev_coords[:,1], s=2, c = 'r') 
        # plt.savefig(fnameplot)
        # plt.close()
        

        # fnameplot = self.folder +  '/recons_initialtopo/'+'scatter_.png' 
        # plt.scatter(self.elev_coords[:,0], self.elev_coords[:,1], s=2)
        # plt.savefig(fnameplot)
        # plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d') 

        # print(self.real_elev_pts.shape, '  self.real_elev_pts')
        # fnameplot = self.folder +  '/recons_initialtopo/'+'scatter3d_elev_.png' 
        # ax.scatter(self.elev_coords[:,0], self.elev_coords[:,1], self.real_elev_pts )
        # plt.savefig(fnameplot)
        # plt.close()
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d') 
        # fnameplot = self.folder +  '/recons_initialtopo/'+'scatter3d_erdp_.png' 
        # ax.scatter(self.erodep_coords[:,0], self.erodep_coords[:,1], self.real_erodep_pts )
        # plt.savefig(fnameplot)
        # plt.close()    

        x = np.arange(0, self.sealevel_data.shape[0], 1)
        fig, ax =  plt.subplots() 
        y = self.sealevel_data[:,1]

        # print(y, ' sea_level')

        fnameplot = self.folder +  '/recons_initialtopo/'+'sealevel_data.png' 
        ax.plot(x, y)
        plt.savefig(fnameplot)
        plt.close()    


        samples = self.samples
        count_list = [] 
        stepsize_vec = np.zeros(self.maxlimits_vec.size)
        span = (self.maxlimits_vec-self.minlimits_vec) 

        for i in range(stepsize_vec.size): # calculate the step size of each of the parameters
            stepsize_vec[i] = self.stepratio_vec[i] * span[i]

        v_proposal = self.vec_parameters # initial param values passed to badlands
        v_current = v_proposal # to give initial value of the chain
 
        #initial_predicted_elev, initial_predicted_erodep, init_pred_erodep_pts_vec, init_pred_elev_pts_vec = self.run_badlands(v_current)
        
        #calc initial likelihood with initial parameters
        [likelihood, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er, rmse_ocean, rmse_elev_ocean] = self.likelihood_func(v_current )

        print('\tinitial likelihood:', likelihood)

        likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
        likeh_list[0,:] = [-10000, -10000] # to avoid prob in calc of 5th and 95th percentile   later
        rmse_elev  = np.ones(samples)  
        rmse_erodep = np.ones(samples)  
        count_list.append(0) # just to count number of accepted for each chain (replica)
        accept_list = np.zeros(samples)
        
        #---------------------------------------
        #now, create memory to save all the accepted tau proposals
        prev_accepted_elev = deepcopy(predicted_elev)
        prev_acpt_erodep_pts = deepcopy(pred_erodep_pts) 
        sum_elev = deepcopy(predicted_elev)
        sum_erodep_pts = deepcopy(pred_erodep_pts)

        #print('time to change')
        burnsamples = int(samples*self.burn_in)
        
        #---------------------------------------
        #now, create memory to save all the accepted   proposals of rain, erod, etc etc, plus likelihood
        pos_param = np.zeros((samples,v_current.size)) 
        list_yslicepred = np.zeros((samples,self.real_elev.shape[0]))  # slice mid y axis  
        list_xslicepred = np.zeros((samples,self.real_elev.shape[1])) # slice mid x axis  
        ymid = int(self.real_elev.shape[1]/2 ) 
        xmid = int(self.real_elev.shape[0]/2)
        list_erodep  = np.zeros((samples,pred_erodep_pts[self.simtime].size))
        list_erodep_time  = np.zeros((samples , self.sim_interval.size , pred_erodep_pts[self.simtime].size))

        init_count = 0
        num_accepted = 0
        num_div = 0 

        initial_samples = 5
        pt_samplesratio = 0.35 # this means pt will be used in begiining and then mcmc with temp of 1 will take place
        pt_samples = int(pt_samplesratio * samples)

        '''with file(('%s/experiment_setting.txt' % (self.folder)),'a') as outfile:
            outfile.write('\nsamples_per_chain:,{0}'.format(self.samples))
            outfile.write('\nburnin:,{0}'.format(self.burn_in))
            outfile.write('\nnum params:,{0}'.format(self.num_param))
            outfile.write('\ninitial_proposed_vec:,{0}'.format(v_proposal))
            outfile.write('\nstepsize_vec:,{0}'.format(stepsize_vec))  
            outfile.write('\nstep_ratio_vec:,{0}'.format(self.stepratio_vec)) 
            outfile.write('\nswap interval:,{0}'.format(self.swap_interval))
            outfile.write('\nsim interval:,{0}'.format(self.sim_interval))
            outfile.write('\nlikelihood_sed (T/F):,{0}'.format(self.check_likelihood_sed))
            outfile.write('\nerodep_coords,elev_coords:,{0}'.format(self.erodep_coords))
            outfile.write('\nsed scaling factor:,{0}'.format(self.sedscalingfactor))'''
        
        start = time.time() 
        self.event.clear()

        for i in range(samples-1):

            print ("Temperature: ", self.temperature, ' Sample: ', i ,"/",samples, pt_samples)

            if i < pt_samples:
                self.adapttemp =  self.temperature #* ratio  #

            if i == pt_samples and init_count ==0: # move to MCMC canonical
                self.adapttemp = 1
                [likelihood, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er, rmse_ocean, rmse_elev_ocean] = self.likelihood_func(v_proposal) 
                init_count = 1

                print('  * adapttemp --------------------------------------- 1 **** ***** ***')

            if self.cov_init and self.use_cov==1:        
                v_p = np.random.normal(size = v_current.shape)
                v_proposal = v_current + np.dot(self.cholesky,v_p)
                # v_proposal = v_current + np.dot(self.cholesky,v_proposal)
            else:
                # Update by perturbing all the  parameters via "random-walk" sampler and check limits

                if i < initial_samples: 
                    v_proposal = np.random.uniform(self.minlimits_vec, self.maxlimits_vec) 
                else:
                    v_proposal =  np.random.normal(v_current,stepsize_vec)

            for j in range(v_current.size):
                if v_proposal[j] > self.maxlimits_vec[j]:
                    v_proposal[j] = v_current[j]
                elif v_proposal[j] < self.minlimits_vec[j]:
                    v_proposal[j] = v_current[j]

            #print(v_proposal)  
            # Passing paramters to calculate likelihood and rmse with new tau
            [likelihood_proposal, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er, rmse_ocean, rmse_elev_ocean] = self.likelihood_func(v_proposal)

            final_predtopo= predicted_elev[self.simtime]
            pred_erodep = pred_erodep_pts[self.simtime]

            # Difference in likelihood from previous accepted proposal
            diff_likelihood = likelihood_proposal - likelihood

            # try:
            #     # print ('diff_likelihood', diff_likelihood)
            #     # print ('math.exp(diff_likelihood)', math.exp(diff_likelihood))
            #     mh_prob = min(1, math.exp(diff_likelihood))
            # except OverflowError as e:
            #     mh_prob = 1

            u = np.log(random.uniform(0,1))
            
            accept_list[i+1] = num_accepted
            likeh_list[i+1,0] = likelihood_proposal

            if u < diff_likelihood: # Accept sample
                # Append sample number to accepted list
                count_list.append(i)            
                
                likelihood = likelihood_proposal
                v_current = v_proposal
                pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector)
                likeh_list[i + 1,1]=likelihood  # contains  all proposal liklihood (accepted and rejected ones)
                list_yslicepred[i+1,:] =  final_predtopo[:, ymid] # slice taken at mid of topography along y axis  
                list_xslicepred[i+1,:]=   final_predtopo[xmid, :]  # slice taken at mid of topography along x axis 
                list_erodep[i+1,:] = pred_erodep
                rmse_elev[i+1,] = avg_rmse_el
                rmse_erodep[i+1,] = avg_rmse_er

                print("Temperature: ", self.temperature, 'Sample', i, 'Likelihood', likelihood , avg_rmse_el, avg_rmse_er, '   --------- ')

                for x in range(self.sim_interval.size): 
                    list_erodep_time[i+1,x, :] = pred_erodep_pts[self.sim_interval[x]]

                num_accepted = num_accepted + 1 
                prev_accepted_elev.update(predicted_elev)

                if i>burnsamples: 
                    
                    for k, v in prev_accepted_elev.items():
                        sum_elev[k] += v 

                    for k, v in pred_erodep_pts.items():
                        sum_erodep_pts[k] += v

                    num_div += 1

            else: # Reject sample
                likeh_list[i + 1, 1]=likeh_list[i,1] 
                pos_param[i+1,:] = pos_param[i,:]
                list_yslicepred[i+1,:] =  list_yslicepred[i,:] 
                list_xslicepred[i+1,:]=   list_xslicepred[i,:]
                list_erodep[i+1,:] = list_erodep[i,:]
                list_erodep_time[i+1,:, :] = list_erodep_time[i,:, :]
                rmse_elev[i+1,] = rmse_elev[i,] 
                rmse_erodep[i+1,] = rmse_erodep[i,]

            
                if i>burnsamples:

                    for k, v in prev_accepted_elev.items():
                        sum_elev[k] += v

                    for k, v in prev_acpt_erodep_pts.items():
                        sum_erodep_pts[k] += v

                    num_div += 1

            if (i >= self.adapt_cov and i % self.adapt_cov == 0 and self.use_cov==1 ) :
                print ('\ncov computed = i ',i, '\n')
                self.computeCovariance(i,pos_param)

            if ( (i+1) % self.swap_interval == 0 ):

                others = np.asarray([likelihood])
                param = np.concatenate([v_current,others,np.asarray([self.temperature])])     

                # paramater placed in queue for swapping between chains
                self.parameter_queue.put(param)
                
                #signal main process to start and start waiting for signal for main
                self.signal_main.set()  
                self.event.clear()         
                self.event.wait()

                result =  self.parameter_queue.get()
                v_current= result[0:v_current.size]     
                #likelihood = result[v_current.size]
         
            save_res =  np.array([i, num_accepted, likelihood, likelihood_proposal, rmse_elev[i+1,], rmse_erodep[i+1,]])  

 

            outfilex = open(('%s/posterior/pos_parameters/stream_chain_%s.txt' % (self.folder, self.temperature)), "a") 
            x = np.array([pos_param[i+1,:]]) 
            np.savetxt(outfilex,x, fmt='%1.8f')  

            outfile1 = open(('%s/posterior/predicted_topo/x_slice/stream_xslice_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile1,np.array([list_xslicepred[i+1,:]]), fmt='%1.2f')  

            outfile2 = open(('%s/posterior/predicted_topo/y_slice/stream_yslice_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile2,np.array([list_yslicepred[i+1,:]]), fmt='%1.2f') 
 
            outfile3 = open(('%s/posterior/stream_res_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile3,np.array([save_res]), fmt='%1.2f')  
 
            outfile4 = open( ('%s/performance/lhood/stream_res_%s.txt' % (self.folder, self.temperature)), "a") 
            np.savetxt(outfile4,np.array([likeh_list[i + 1,0]]), fmt='%1.2f') 

            outfile5 = open( ('%s/performance/accept/stream_res_%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile5,np.array([accept_list[i+1]]), fmt='%1.2f')

            outfile6 = open(  ('%s/performance/rmse_erdp/stream_res_%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile6,np.array([rmse_erodep[i+1,]]), fmt='%1.2f')

            outfile7 = open( ('%s/performance/rmse_elev/stream_res_%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile7,np.array([rmse_elev[i+1,]]), fmt='%1.2f')

            outfile8 = open( ('%s/performance/rmse_ocean/stream_res_ocean%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile8, np.array([rmse_elev_ocean]), fmt='%1.2f', newline='\n') 

            outfile9 = open( ('%s/performance/rmse_ocean/stream_res_ocean_t%s.txt' % (self.folder, self.temperature)), "a")  
            np.savetxt(outfile9, np.array([rmse_ocean]), fmt='%1.2f', newline='\n')

            temp = list_erodep_time[i+1,-1,:]  
            temp = np.reshape(temp, temp.shape[0]*1) 
 
            outfile10 = open( (self.folder + '/posterior/predicted_topo/sed/chain_' + str(self.temperature) + '.txt'), "a") 
            np.savetxt(outfile10, np.array([temp]), fmt='%1.2f') 

        others = np.asarray([ likelihood])
        param = np.concatenate([v_current,others,np.asarray([self.temperature])])  

        self.parameter_queue.put(param) 
        self.signal_main.set()  

        accepted_count =  len(count_list) 
        accept_ratio = accepted_count / (samples * 1.0) * 100

        print(accept_ratio, ' accept_ratio ')

        for k, v in sum_elev.items():
            sum_elev[k] = np.divide(sum_elev[k], num_div)
            mean_pred_elevation = sum_elev[k]

            sum_erodep_pts[k] = np.divide(sum_erodep_pts[k], num_div)
            mean_pred_erodep_pnts = sum_erodep_pts[k]

            file_name = self.folder + '/posterior/predicted_topo/topo/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
            np.savetxt(file_name, mean_pred_elevation, fmt='%.2f')

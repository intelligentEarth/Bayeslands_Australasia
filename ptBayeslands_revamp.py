#Main Contributers:   Rohitash Chandra and Danial Azam  Email: c.rohitash@gmail.com 

# Bayeslands II: Parallel tempering for multi-core systems - Badlands

from __future__ import print_function, division
import os
import shutil
import sys
import random
import time
import operator
import math 
import copy
import fnmatch
import collections
import numpy as np
import matplotlib as mpl
import re 
import pandas as pd
import h5py
import json
mpl.use('Agg')
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import multiprocessing
import itertools
import chart_studio
import chart_studio.plotly as py
import pandas
import argparse
import pandas as pd
# import seaborn as sns
import scipy.ndimage as ndimage
#plotly.offline.init_notebook_mode()
from plotly.graph_objs import *
from pylab import rcParams
from copy import deepcopy 
from pylab import rcParams
from scipy import special
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from badlands.model import Model as badlandsModel
import badlands 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.ndimage import filters 
from scipy.ndimage import gaussian_filter
from problem_setup import problem_setup
import subprocess
from dbfpy3 import dbf
# import dbfpy.dbf as dbf
# import dbf
# from process_Uplift import edit_Uplift, process_Uplift
import pickle

#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')

parser.add_argument('-p','--problem', help='Problem Number 1-crater-fast,2-crater,3-etopo-fast,4-etopo,5-null,6-mountain', required=True,   dest="problem",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=10,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap interval', dest="swap_interval",default= 2,type=int)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)  
parser.add_argument('-rain_intervals','--rain_intervals', help='rain_intervals', dest="rain_intervals",default=4,type=int)
parser.add_argument('-epsilon','--epsilon', help='epsilon for inital topo', dest="epsilon",default=0.5,type=float)
parser.add_argument('-cov','--covariance', help='flag for covariance', dest="covariance",default=0,type=int)
parser.add_argument('-inittopo', '--initialtopo', help='flag for init topo inference', dest="inittopo", default=1, type=int)
parser.add_argument('-uplift', '--uplift', help='flag for uplift inference', dest="uplift", default=1, type=int)

args = parser.parse_args()
    
#parameters for Parallel Tempering
problem = args.problem
samples = args.samples 
num_chains = args.num_chains
swap_interval = args.swap_interval
burn_in=args.burn_in
#maxtemp = int(num_chains * 5)/args.mt_val
maxtemp =   args.mt_val  
num_successive_topo = 4
pt_samples = args.pt_samples
epsilon = args.epsilon
rain_intervals = args.rain_intervals
covariance = args.covariance
inittopo = args.inittopo
uplift = args.uplift

method = 1 # type of formaltion for inittopo construction (Method 1 showed better results than Method 2)

class ptReplica(multiprocessing.Process):
    
    def __init__(self, ID, num_param, vec_parameters, sea_level, ocean_t,  inittopo_expertknow, rain_region, rain_time, len_grid, wid_grid, minlimits_vec, maxlimits_vec, stepratio_vec,   check_likelihood_sed ,  swap_interval, sim_interval, simtime, samples, init_elev, real_elev,  real_erodep_pts, real_elev_pts, erodep_coords,elev_coords, filename, xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in, inittopo_estimated, covariance, Bayes_inittopoknowledge):

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

    def init_show(self, zData, fname, replica_id): 
 
        fig = plt.figure()
        im = plt.imshow(zData, cmap='hot', interpolation='nearest')
        plt.colorbar(im)
        plt.savefig(self.folder + fname+ str(int(replica_id))+'.png')
        plt.close()
 
    def process_inittopoGMT(self, inittopo_vec):

        self.edit_DBF(inittopo_vec)

        bashcommand = 'sh ptopo_150.sh %s' %(int(self.ID))

        process = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)

        output, error = process.communicate()

    def edit_DBF(self, inittopo_vec):  #edit shape file for init topo reconstruction 

        # expert_know = np.loadtxt('init_topo_polygon/dbf_polygon.txt')
        expert_know = np.loadtxt('init_topo_polygon/polygon_elev.txt')
        
        print(expert_know, ' is loaded expert knowledge for Chain %s' %(self.ID))
     
        # DBFPY IMPLEMENTATION
        # print(' SHAPE OF INITOPO VEC', inittopo_vec.shape)
        # print(' SHAPE OF EXPERT KNOWLEDGE', expert_know.shape)
        
        db = dbf.Dbf("init_topo_polygon/data/%s/Paleotopo_P400.dbf"%(self.ID))
    
        x = 0
    
        for i,rec in enumerate(db):
            
            if rec[0] == "Uplands":
                rec["ELEVATION"] = (inittopo_vec[x]*(0.25*1500)) + expert_know[i]
                x = x + 1
                rec.store()
                del rec
            elif rec[0] == "Land unclassified":
                rec["ELEVATION"] = (inittopo_vec[x]*(0.25*700)) + expert_know[i]      
                x = x + 1
                rec.store()
                del rec
            elif rec[0] == "Land":
                rec["ELEVATION"] = (inittopo_vec[x]*(0.25*600)) + expert_know[i]
                x = x + 1
                rec.store()
                del rec
            elif rec[0] == "Land erosional":
                rec["ELEVATION"] = (inittopo_vec[x]*(0.25*1500)) + expert_know[i]
                x = x + 1
                rec.store()
                del rec
            else:
                pass
                # Do Nothing
        db.close()

        return

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

        print(coeff, ' coeff -----------------')

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
        plt.clf()    

        proposed_sealevel = np.vstack([timeframes, yhat])

        return proposed_sealevel

    def smooth(self, y, box_pts):
        #https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
        #print(y.shape, y, ' ++ y ')
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def run_badlands(self, input_vector):
        #Runs a badlands model with the specified inputs

        print(self.real_elev.shape, ' real evel sh')


        print(self.real_elev_pts.shape, ' real evel pt sh')
 
        rain_regiontime = self.rain_region * self.rain_time # number of parameters for rain based on  region and time 

        #Create a badlands model instance
        model = badlandsModel()

        if uplift == 1:
            xml_id = int(self.ID)
        else:
            xml_id = 0

        print(xml_id, input_vector[11], ' xml_id  input_vector[11]')

        xmlinput = self.input[xml_id]

        #----------------------------------------------------------------
        # Load the XmL input file
        model.load_xml(str(self.run_nb), xmlinput, verbose =False, muted = True)

        num_sealevel_coef = 10

        if inittopo == 1:

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
 

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[rain_regiontime]  
        model.flow.erodibility.fill(input_vector[rain_regiontime ] )

        # Adjust m and n values
        model.input.SPLm = input_vector[rain_regiontime+1]  
        model.input.SPLn = input_vector[rain_regiontime+2] 

        #Check if it is the etopo extended problem
        #if problem == 4 or problem == 3:  # will work for more parameters
        model.input.CDm = input_vector[rain_regiontime+3] # submarine diffusion
        model.input.CDa = input_vector[rain_regiontime+4] # aerial diffusion

        if problem != 1:
            model.slp_cr = input_vector[rain_regiontime+5]
            model.perc_dep = input_vector[rain_regiontime+6]
            model.input.criver = input_vector[rain_regiontime+7]
            model.input.elasticH = input_vector[rain_regiontime+8]
            model.input.diffnb = input_vector[rain_regiontime+9]
            model.input.diffprop = input_vector[rain_regiontime+10]


        sealevel_coeff = input_vector[rain_regiontime+12 : rain_regiontime+12+ num_sealevel_coef] 

        print(sealevel_coeff, ' sealevel_coeff ')
 
 

        model.input.curve = self.process_sealevel(sealevel_coeff)
 
        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()
        elev_pts_vec = collections.OrderedDict()




        model.run_to_time(-1.489999e08, muted = True)
        elev_, erodep_ = interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff) 

        self.init_show(elev_, '/pred_plots/GMTinit_', self.ID )   

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]
            model.run_to_time(self.simtime, muted = True)

            elev, erodep = interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

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

        likelihood_elev_ocean = 0
        rmse_ocean = np.zeros(self.sim_interval.size)
        pred_topo_presentday = pred_elev_vec[self.simtime] 
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

            # print('\n sim_interval[i] ', sim_interval[i], ' matches : ', matches ,'  non matches : ', non_matches, 'percentage non match', (non_matches/p_elev_ocean.size)*100)
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

            tausq_ocean = np.sum(np.square(p_elev_ocean - r_elev_ocean))/self.real_elev.size  
            rmse_ocean[i] = tausq_ocean
            likelihood_elev_ocean  += np.sum(-0.5 * np.log(2 * math.pi * tausq_ocean) - 0.5 * np.square(p_elev_ocean - r_elev_ocean) /  tausq_ocean )

        '''fig = plt.figure()
        plt.imshow(self.real_erodep_pts, cmap='hot', interpolation='nearest')
        plt.savefig(self.folder +'/realtime_plots/' + str(self.ID) + '/'+ str(self.sim_interval[i]) +'real_erodep.png')
        plt.close()'''
        
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

        real_elev_filtered = np.where((self.real_elev>0) & (self.real_elev<600), self.real_elev, 0)  
        pred_elev_filtered = np.where((pred_elev>0) & (pred_elev<600), pred_elev, 0)

        diff = pred_elev_filtered  - real_elev_filtered
        count = np.count_nonzero(diff) 

        tau_elev =  np.sum(np.square(diff)) / count

        likelihood_elev  = np.sum(-0.5 * np.log(2 * math.pi * tau_elev ) - 0.5 * np.square(diff) / tau_elev )
        likelihood_erodep  = np.sum(-0.5 * np.log(2 * math.pi * tau_erodep ) - 0.5 * np.square(erdep_predicted - self.real_erodep_pts) / tau_erodep ) # only considers point or core of erodep    

        likelihood_ = likelihood_erodep + (likelihood_elev_ocean/4)
        #rmse_ocean = 0
        rmse_elev = np.sqrt(tau_elev)
        rmse_elev_ocean = np.average(rmse_ocean)
        rmse_erodep = np.sqrt(tau_erodep) 
        rmse_elev_pts = np.sqrt(tau_elev) 

        likelihood = likelihood_*(1.0/self.adapttemp)

        pred_topo_presentday = pred_elev_vec[self.simtime]
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

        if problem ==2: 
            fnameplot = self.folder +  '/recons_initialtopo/'+'scatter_erodep_.png' 
            plt.scatter(self.erodep_coords[:,0], self.erodep_coords[:,1], s=2, c = 'b')
            plt.scatter(self.elev_coords[:,0], self.elev_coords[:,1], s=2, c = 'r') 
            plt.savefig(fnameplot)
            plt.clf()
            

            fnameplot = self.folder +  '/recons_initialtopo/'+'scatter_.png' 
            plt.scatter(self.elev_coords[:,0], self.elev_coords[:,1], s=2)
            plt.savefig(fnameplot)
            plt.clf()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') 

            print(self.real_elev_pts.shape, '  self.real_elev_pts')
            fnameplot = self.folder +  '/recons_initialtopo/'+'scatter3d_elev_.png' 
            ax.scatter(self.elev_coords[:,0], self.elev_coords[:,1], self.real_elev_pts )
            plt.savefig(fnameplot)
            plt.clf()
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') 
            fnameplot = self.folder +  '/recons_initialtopo/'+'scatter3d_erdp_.png' 
            ax.scatter(self.erodep_coords[:,0], self.erodep_coords[:,1], self.real_erodep_pts )
            plt.savefig(fnameplot)
            plt.clf()    
 
            x = np.arange(0, self.sealevel_data.shape[0], 1)
            fig, ax =  plt.subplots() 
            y = self.sealevel_data[:,1]

            print(y, ' sea_level')

            fnameplot = self.folder +  '/recons_initialtopo/'+'sealevel_data.png' 
            ax.plot(x, y)
            plt.savefig(fnameplot)
            plt.clf()    

            #self.sealevel_data   
             

     
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

            try:
                # print ('diff_likelihood', diff_likelihood)
                # print ('math.exp(diff_likelihood)', math.exp(diff_likelihood))
                mh_prob = min(1, math.exp(diff_likelihood))
            except OverflowError as e:
                mh_prob = 1

            u = random.uniform(0,1)
            
            accept_list[i+1] = num_accepted
            likeh_list[i+1,0] = likelihood_proposal

            if u < mh_prob: # Accept sample
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
 
class ParallelTempering:

    def __init__(self,  vec_parameters, sea_level, ocean_t, inittopo_expertknow, rain_region, rain_time,  len_grid,  wid_grid, num_chains, maxtemp,NumSample,swap_interval, fname, realvalues_vec, num_param, init_elev, real_elev, erodep_pts, elev_pts, erodep_coords,elev_coords, simtime, siminterval, resolu_factor, run_nb, inputxml,inittopo_estimated, covariance, Bayes_inittopoknowledge):
        self.swap_interval = swap_interval
        self.folder = fname
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 
        self.real_erodep_pts  = erodep_pts
        self.real_elev_pts = elev_pts
        self.real_elev = real_elev
        self.init_elev = init_elev
        self.ocean_t = ocean_t
        self.resolu_factor =  resolu_factor
        self.num_param = num_param
        self.erodep_coords  = erodep_coords 
        self.elev_coords =  elev_coords
        self.simtime = simtime
        self.sim_interval = siminterval
        self.run_nb =run_nb 
        self.xmlinput = inputxml
        self.vec_parameters = vec_parameters
        self.realvalues  =  realvalues_vec 

        self.sealevel_data = sea_level

        # create queues for transfer of parameters between process chain
        #self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()  
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        # two ways events are used to synchronize chains
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        #self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        self.geometric =  True
        self.total_swap_proposals = 0
        self.rain_region = rain_region  
        self.rain_time = rain_time
        self.len_grid = len_grid
        self.wid_grid = wid_grid
        self.inittopo_expertknow =  inittopo_expertknow 
        self.inittopo_estimated = inittopo_estimated
        self.Bayes_inittopoknowledge = Bayes_inittopoknowledge

        self.covariance = covariance

    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        
        """
        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                        1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                        1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                        1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                        1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                        1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                        1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                        1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                        1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                        1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                        1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                        1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                        1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                        1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                        1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                        1.26579, 1.26424, 1.26271, 1.26121,
                        1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        #Geometric Spacing
        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)      
            for i in range(0, self.num_chains):         
                self.temperatures.append(np.inf if betas[i] is 0 else 1.0/betas[i])
                print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp /self.num_chains)
            temp = 1
            print("Temperatures...")
            for i in xrange(0, self.num_chains):            
                self.temperatures.append(temp)
                temp += tmpr_rate
                print(self.temperatures[i])

    def initialize_chains (self,  minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed,   burn_in):
        self.burn_in = burn_in
        
        self.assign_temperatures()
        xml_list = ['Examples/australia/AUSP1307.xml','Examples/australia/AUSP1309.xml', 'Examples/australia/AUSP1310.xml',
        'Examples/australia/AUSP1311.xml','Examples/australia/AUSP1312.xml', 'Examples/australia/AUSP1313.xml', 'Examples/australia/AUSP1314.xml',
        'Examples/australia/AUSP1315.xml']
        
        for i in range(0, self.num_chains):
            self.vec_parameters =  np.random.uniform(minlimits_vec, maxlimits_vec)
            self.xmlinput = xml_list     
            self.chains.append(ptReplica(i, self.num_param, self.vec_parameters, self.sealevel_data, self.ocean_t, self.inittopo_expertknow, self.rain_region, self.rain_time, self.len_grid, self.wid_grid, minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed ,self.swap_interval, self.sim_interval,   self.simtime, self.NumSamples, self.init_elev, self.real_elev,   self.real_erodep_pts, self.real_elev_pts, self.erodep_coords,self.elev_coords, self.folder, self.xmlinput,  self.run_nb,self.temperatures[i], self.parameter_queue[i],self.event[i], self.wait_chain[i],burn_in, self.inittopo_estimated, self.covariance, self.Bayes_inittopoknowledge))

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        # if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
            param1 = parameter_queue_1.get()
            param2 = parameter_queue_2.get()

            w1 = param1[0:self.num_param] 
            lhood1 = param1[self.num_param+1]
            T1 = param1[self.num_param+1]
            w2 = param2[0:self.num_param] 
            lhood2 = param2[self.num_param+1]
            T2 = param2[self.num_param+1]

            try:
                swap_proposal =  min(1,0.5*np.exp(min(709, lhood2 - lhood1)))
            except OverflowError:
                swap_proposal = 1
            u = np.random.uniform(0,1)
            swapped = False
            if u < swap_proposal: 
                self.total_swap_proposals += 1
                self.num_swap += 1
                param_temp =  param1
                param1 = param2
                param2 = param_temp
                swapped = True
            else:
                swapped = False
                self.total_swap_proposals += 1
            return param1, param2,swapped

    def run_chains (self ):
    
        swap_proposal = np.ones(self.num_chains-1) 
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))  
        lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1
        number_exchange = np.zeros(self.num_chains)
        filen = open(self.folder + '/num_exchange.txt', 'a')
        #RUN MCMC CHAINS
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0,self.num_chains):        
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()
        #SWAP PROCEDURE

        swaps_appected_main =0
        total_swaps_main =0
        for i in range(int(self.NumSamples/self.swap_interval)):
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count+=1
                    self.wait_chain[index].set()
                    print(str(self.chains[index].temperature) +" Dead")

            if count == self.num_chains:
                break
            print("Waiting")
            timeout_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                print("Skipping the swap!")
                continue
            print("Event occured")
            for index in range(0,self.num_chains-1):
                print('starting swap')
                param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index+1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_appected_main += 1
                    total_swaps_main += 1
            for index in range (self.num_chains):
                    self.event[index].set()
                    self.wait_chain[index].clear()

        print("Joining processes")

        #JOIN THEM TO MAIN PROCESS
        for index in range(0,self.num_chains):
            self.chains[index].join()
        self.chain_queue.join()

        print(number_exchange, 'num_exchange, process ended')

        combined_topo, accept, pred_topofinal, combined_topo = self.show_results('chain_')
        
        for i in range(self.sim_interval.size): 

            self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=combined_topo[i,:,:], title='Predicted Topography ', time_frame=self.sim_interval[i],  filename= 'mean')
        
        swap_perc = self.num_swap  #*100/self.total_swap_proposals  

        simulated_topofinal = combined_topo[self.sim_interval.size-1,:,:]

        self.full_crosssection(simulated_topofinal, self.real_elev) 

        return (pred_topofinal, swap_perc, accept)

    def full_crosssection(self,  simulated_topo, real_elev):

        ymid = int( real_elev.shape[1]/2)  

        x = np.linspace(0, real_elev.shape[0], num=real_elev.shape[0])
        x_m = np.arange(0,real_elev.shape[0], 10)

        for i in x_m:
            xmid = i 

            real = real_elev[0:real_elev.shape[0], i]  
            pred = simulated_topo[0:real_elev.shape[0], i]
            size = 15

            plt.tick_params(labelsize=size)
            params = {'legend.fontsize': size, 'legend.handlelength': 2}
            plt.rcParams.update(params)
            plt.plot(x, real, label='Ground Truth') 
            plt.plot(x, pred, label='Badlands Pred.') 
            plt.grid(alpha=0.75)
            plt.legend(loc='best')  
            plt.title("Topography cross section   ", fontsize = size)
            plt.xlabel(' Distance (x 50 km)  ', fontsize = size)
            plt.ylabel(' Height (m)', fontsize = size)
            plt.tight_layout()
              
            plt.savefig(self.folder+'/cross_section/'+str(i)+'_cross-sec.pdf')
            plt.clf()

    # Merge different MCMC chains y stacking them on top of each other
    def show_results(self, filename):

        burnin = int(self.NumSamples * self.burn_in)
        accept_percent = np.zeros((self.num_chains, 1)) 
        topo  = self.real_elev
        replica_topo = np.zeros((self.sim_interval.size, self.num_chains, topo.shape[0], topo.shape[1])) #3D
        combined_topo = np.zeros(( self.sim_interval.size, topo.shape[0], topo.shape[1]))

        for i in range(self.num_chains):
            for j in range(self.sim_interval.size):

                file_name = self.folder+'/posterior/predicted_topo/topo/chain_'+str(self.sim_interval[j])+'_'+ str(self.temperatures[i])+ '.txt'
                dat_topo = np.loadtxt(file_name)
                replica_topo[j,i,:,:] = dat_topo

        for j in range(self.sim_interval.size):
            for i in range(self.num_chains):
                combined_topo[j,:,:] += replica_topo[j,i,:,:]  
            combined_topo[j,:,:] = combined_topo[j,:,:]/self.num_chains
            # dx = combined_erodep[j,:,:,:].transpose(2,0,1).reshape(self.real_erodep_pts.shape[1],-1)
            # timespan_erodep[j,:,:] = dx.T

        accept = np.sum(accept_percent)/self.num_chains

        pred_topofinal = combined_topo[-1,:,:] # get the last mean pedicted topo to calculate mean squared error loss 

        return  combined_topo, accept, pred_topofinal, combined_topo
        #---------------------------------------

    def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

        # if zmin == None:
        #     zmin =  zData.min()

        # if zmax == None:
        #     zmax =  zData.max()

        # tickvals= [0,50,75,-50]

        # xx = (np.linspace(0, int(zData.shape[0]* self.resolu_factor), num=int(zData.shape[0]/10 ))) 
        # yy = (np.linspace(0, int(zData.shape[1] * self.resolu_factor), num=int(zData.shape[1]/10) )) 

        # xx = np.around(xx, decimals=0)
        # yy = np.around(yy, decimals=0)
        # print (xx,' xx')
        # print (yy,' yy')

        # axislabelsize = 20

        # data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YlGnBu')])

        # layout = Layout(  autosize=True, width=width, height=height,scene=Scene(
        #             zaxis=ZAxis(title = 'Elev.   ', range=[zmin,zmax], autorange=False, nticks=5, gridcolor='rgb(255, 255, 255)',
        #                         gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
        #             xaxis=XAxis(title = 'x-axis  ',  tickvals= xx,      gridcolor='rgb(255, 255, 255)', gridwidth=2,
        #                         zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
        #             yaxis=YAxis(title = 'y-axis  ', tickvals= yy,    gridcolor='rgb(255, 255, 255)', gridwidth=2,
        #                         zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
        #             bgcolor="rgb(244, 244, 248)"
        #         )
        #     )

        # fig = Figure(data=data, layout=layout) 
        # graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.folder +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.html', validate=False)

        fname = self.folder + '/pred_plots'+'/pred_'+filename+'_'+str(time_frame)+ '_.pdf' 
        elev_data = np.reshape(zData, zData.shape[0] * zData.shape[1] )   
        hist, bin_edges = np.histogram(elev_data, density=True)

        size = 15 
        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.hist(elev_data, bins='auto')  

        #plt.title("Topography")  
        plt.xlabel('Elevation (m)', fontsize = size)
        plt.ylabel('Frequency', fontsize = size)
        plt.grid(alpha=0.75)


        plt.tight_layout()  
        plt.savefig(fname )
        plt.clf()

        fname = self.folder + '/pred_plots'+'/pred_'+filename+'_'+str(time_frame)+ '_.txt'
        np.savetxt(fname, zData, fmt='%1.2f')
# class  above this line -------------------------------------------------------------------------------------------------------

def interpolateArray(coords=None, z=None, dz=None):
    """
    Interpolate the irregular spaced dataset from badlands on a regular grid.
    """
    x, y = np.hsplit(coords, 2)
    dx = (x[1]-x[0])[0]

    if problem == 1:
        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)
    else:
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

def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():

    random.seed(time.time()) 

    (problemfolder, xmlinput, simtime, resolu_factor, sea_level, init_elev, groundtruth_elev, groundtruth_erodep,
    groundtruth_erodep_pts, groundtruth_elev_pts, res_summaryfile, inittopo_expertknow, len_grid, wid_grid, likelihood_sediment, rain_min, rain_max, rain_regiongrid, minlimits_others,
    maxlimits_others, stepsize_ratio, erodep_coords,elev_coords,inittopo_estimated, vec_parameters, minlimits_vec,
    maxlimits_vec) = problem_setup(problem)



    print('\n\ngroundtruth_elev_pts[0]',groundtruth_elev_pts[0],'\n\n')

    rain_timescale = rain_intervals  # to show climate change 

    true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
    stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
    num_param = vec_parameters.size
 
    Bayes_inittopoknowledge = False # True means that you are using revised expert knowledge. False means you are making adjustment to expert knowledge  # NOT USED ANYMORE

    fname = ""
    run_nb = 0
    while os.path.exists( problemfolder+ 'results_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists( problemfolder+ 'results_%s' % (run_nb)):
        os.makedirs( problemfolder+ 'results_%s' % (run_nb))
        fname = ( problemfolder+ 'results_%s' % (run_nb))

    #fname = ('sampleresults')

    make_directory((fname + '/posterior/pos_parameters')) 
    make_directory((fname + '/recons_initialtopo')) 
    make_directory((fname + '/pos_plots')) 
    make_directory((fname + '/cross_section')) 
    make_directory((fname + '/sediment_plots')) 
    make_directory((fname + '/posterior/predicted_topo/topo'))  
    make_directory((fname + '/posterior/predicted_topo/sed'))  
    make_directory((fname + '/posterior/predicted_topo/x_slice'))
    make_directory((fname + '/posterior/predicted_topo/y_slice'))
    make_directory((fname + '/posterior/posterior/predicted_erodep')) 
    make_directory((fname + '/pred_plots'))
    for i in range(10):
        make_directory(fname + '/realtime_plots/%s' %(i))
        make_directory(fname + '/realtime_data/%s' %(i))
        
    make_directory((fname + '/strat_plots'))
    make_directory((fname + '/sed_visual'))
    make_directory((fname + '/performance/lhood'))
    make_directory((fname + '/performance/accept'))
    make_directory((fname + '/performance/rmse_erdp'))
    make_directory((fname + '/performance/rmse_elev'))
    make_directory((fname + '/performance/rmse_ocean'))

    print ('\n\nfolder --',np.array([fname]), '\n\n')
    np.savetxt('foldername.txt', np.array([fname]), fmt="%s")

    run_nb_str =  'results_' + str(run_nb)
    timer_start = time.time()
    
    
    ### 149 MA
    if problem ==1:
        num_successive_topo = 4 
        sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
        filename_ocean = np.array([0, 5 , 25 , 30, 40 ])

    else:
        sim_interval = np.array([0, -5.0e06 , -25.0e06, -30.0e06,  -40.0e06, -50.0e06 , -75.0e06 , -100.0e06,  -115.0e06, -125.0e06, -1.40e08,  -1.49e08])
        filename_ocean = np.array([0, 5 , 25 , 30, 40, 50, 75, 100, 115,  125, 140, 149])

    ### 1 MA 
    # sim_interval = np.array([0, -5.0e04 , -25.0e04, -50.0e04 , -75.0e04 , -100.0e04, -125.0e04, -1.49e06])
    # filename_ocean = np.array([0, 5, 25, 50, 75, 100, 125, 149])
    #sim_interval = np.array([0, -5.0e04 , -25.0e04, -50.0e04 , -75.0e04 , -100.0e04, -125.0e04, -1.49e05,  -5.49e05,  -0.49e06,  -1.19e06,  -1.49e06])
    #filename_ocean = np.array([0, 5, 25, 50, 75, 100, 125, 149])
 

    print ('Simulation time interval before',sim_interval)
    if simtime < 0:
        sim_interval = sim_interval[::-1]
        filename_ocean = filename_ocean[::-1]

    print("Simulation time interval", sim_interval)
    print()

    ocean_t = np.zeros((sim_interval.size,groundtruth_elev.shape[0], groundtruth_elev.shape[1]))

    if problem ==2: 
        for i, val in enumerate(filename_ocean): 
            temp = np.loadtxt(problemfolder+ '/data/ocean/marine_%s.txt' %(val))
            ocean_t[i,:,:] = temp

    # print(ocean_t, 'ocean_t')

    #-------------------------------------------------------------------------------------
    #Create A a Patratellel Tempring object instance 
    #-------------------------------------------------------------------------------------
    pt = ParallelTempering(vec_parameters, sea_level, ocean_t, inittopo_expertknow, rain_regiongrid, rain_timescale, len_grid,  wid_grid, num_chains, maxtemp, samples,swap_interval,fname, true_parameter_vec, num_param  , init_elev, groundtruth_elev,  groundtruth_erodep_pts , groundtruth_elev_pts,  erodep_coords,elev_coords, simtime, sim_interval, resolu_factor, run_nb_str, xmlinput, inittopo_estimated, covariance, Bayes_inittopoknowledge)
    
    #-------------------------------------------------------------------------------------
    # intialize the MCMC chains
    #-------------------------------------------------------------------------------------
    pt.initialize_chains(minlimits_vec, maxlimits_vec, stepratio_vec, likelihood_sediment,   burn_in)

    #-------------------------------------------------------------------------------------
    #run the chains in a sequence in ascending order
    #-------------------------------------------------------------------------------------
    pred_topofinal, swap_perc, accept  = pt.run_chains()

    print('sucessfully sampled') 
    timer_end = time.time()  

    #stop()

if __name__ == "__main__": main()

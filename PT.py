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
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import multiprocessing
import itertools
import chart_studio
import chart_studio.plotly as py
import pandas
import pickle
import argparse
import subprocess
import pandas as pd
import scipy.ndimage as ndimage
from plotly.graph_objs import *
from pylab import rcParams
from copy import deepcopy 
from scipy import special
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.ndimage import filters 
from scipy.ndimage import gaussian_filter
from badlands.model import Model as badlandsModel

from dbfpy3 import dbf
import PtReplica as Chain
from GenUplift import edit_Uplift, process_Uplift
from GenInitTopo import process_inittopoGMT, edit_DBF
from ProblemSetup import ProblemSetup

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

class ParallelTempering:

    def __init__(self, vec_parameters, sea_level, ocean_t, inittopo_expertknow, rain_region, rain_time,  len_grid,  wid_grid, num_chains, maxtemp,NumSample,swap_interval, fname, realvalues_vec, num_param, init_elev, real_elev, erodep_pts, elev_pts, erodep_coords,elev_coords, simtime, siminterval, resolu_factor, run_nb, inputxml,inittopo_estimated, covariance, Bayes_inittopoknowledge):
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
            for i in range(0, self.num_chains):            
                self.temperatures.append(temp)
                temp += tmpr_rate
                print(self.temperatures[i])

    def initialize_chains (self,  minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed,   burn_in):
        self.burn_in = burn_in
        
        self.assign_temperatures()
        # xml_list = ['Examples/australia/AUSB004.xml','Examples/australia/AUSP1307.xml', 'Examples/australia/AUSP1310.xml',
        # 'Examples/australia/AUSP1311.xml','Examples/australia/AUSP1312.xml', 'Examples/australia/AUSP1313.xml', 'Examples/australia/AUSP1314.xml',
        # 'Examples/australia/AUSP1315.xml']
        xml_list = ['Examples/australia_gda94/AUSB004.xml','Examples/australia_gda94/AUSB004.xml','Examples/australia_gda94/AUSB004.xml','Examples/australia_gda94/AUSB004.xml'
        ,'Examples/australia_gda94/AUSB004.xml','Examples/australia_gda94/AUSB004.xml','Examples/australia_gda94/AUSB004.xml','Examples/australia_gda94/AUSB004.xml']
        
        for i in range(0, self.num_chains):
            self.vec_parameters =  np.random.uniform(minlimits_vec, maxlimits_vec)
            self.xmlinput = xml_list     
            self.chains.append(Chain.PtReplica(i, self.num_param, self.vec_parameters, self.sealevel_data, self.ocean_t, self.inittopo_expertknow, self.rain_region, self.rain_time, self.len_grid, self.wid_grid, minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed ,self.swap_interval, self.sim_interval,   self.simtime, self.NumSamples, self.init_elev, self.real_elev,   self.real_erodep_pts, self.real_elev_pts, self.erodep_coords,self.elev_coords, self.folder, self.xmlinput,  self.run_nb,self.temperatures[i], self.parameter_queue[i],self.event[i], self.wait_chain[i],burn_in, self.inittopo_estimated, self.covariance, self.Bayes_inittopoknowledge, uplift,inittopo))

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

def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():

    random.seed(time.time()) 

    (problemfolder, xmlinput, simtime, resolu_factor, sea_level, init_elev, groundtruth_elev, groundtruth_erodep,
    groundtruth_erodep_pts, groundtruth_elev_pts, res_summaryfile, inittopo_expertknow, len_grid, wid_grid, likelihood_sediment, rain_min, rain_max, rain_regiongrid, minlimits_others,
    maxlimits_others, stepsize_ratio, erodep_coords,elev_coords,inittopo_estimated, vec_parameters, minlimits_vec,
    maxlimits_vec) = ProblemSetup()

    print('\n\ngroundtruth_elev_pts[0]',groundtruth_elev_pts[0],'\n\n')

    print(groundtruth_elev.shape)
    print(groundtruth_erodep.shape)
    print(groundtruth_erodep_pts.shape)
    print(groundtruth_elev_pts.shape)

    exit

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
    
    # if problem ==1:
    #     num_successive_topo = 4 
    #     sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
    #     filename_ocean = np.array([0, 5 , 25 , 30, 40 ])

    # else:
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

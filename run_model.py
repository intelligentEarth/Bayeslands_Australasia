from __future__ import print_function, division
from badlands.model import Model as badlandsModel
import numpy as np
from scipy.spatial import cKDTree
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math, csv
def interpolateArray(coords=None, z=None, dz=None):
    """
    Interpolate the irregular spaced dataset from badlands on a regular grid.
    """
    x, y = np.hsplit(coords, 2)
    dx = (x[1]-x[0])[0]

    nx = int(((x.max() - x.min())/dx+1) - 2)
    ny = int(((y.max() - y.min())/dx+1) - 2)
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

def main():
    model = badlandsModel()
    model.load_xml('Output','Examples/australia_gda94/AUSB004.xml', muted = False)
    model.run_to_time(0, muted = False)

    # input_vector = np.loadtxt('variables.txt')
    # rain_regiontime = 4
    # num_sealevel_coef = 10


    # model.force.rainVal[:] = input_vector[0:rain_regiontime] 

    # # Adjust erodibility based on given parameter
    # model.input.SPLero = input_vector[rain_regiontime]  
    # model.flow.erodibility.fill(input_vector[rain_regiontime])

    # # Adjust m and n values
    # model.input.SPLm = input_vector[rain_regiontime+1]  
    # model.input.SPLn = input_vector[rain_regiontime+2] 

    # #Check if it is the etopo extended problem
    # #if problem == 4 or problem == 3:  # will work for more parameters
    # model.input.CDm = input_vector[rain_regiontime+3] # submarine diffusion
    # model.input.CDa = input_vector[rain_regiontime+4] # aerial diffusion

    # model.slp_cr = input_vector[rain_regiontime+5]
    # model.perc_dep = input_vector[rain_regiontime+6]
    # model.input.criver = input_vector[rain_regiontime+7]
    # model.input.elasticH = input_vector[rain_regiontime+8]
    # model.input.diffnb = input_vector[rain_regiontime+9]
    # model.input.diffprop = input_vector[rain_regiontime+10]

    # sealevel_coeff = input_vector[rain_regiontime+12 : rain_regiontime+12+ num_sealevel_coef] 


    elev, erdp = interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)
    np.savetxt("Examples/australia/data/testrun_final_elev.txt", elev)
    np.savetxt("Examples/australia/data/testrun_final_erdp.txt", erdp)

if __name__ == "__main__": main()
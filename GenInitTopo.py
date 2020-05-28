
import subprocess
import os
import pandas as pd
import numpy as np
from dbfpy3 import dbf

def process_inittopoGMT(self, inittopo_vec):

    self.edit_DBF(inittopo_vec)

    bashcommand = 'sh ptopo_150.sh %s' %(int(self.ID))

    process = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)

    output, error = process.communicate()

    return

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

def main():
    inittopo_vec = np.ones((51,1))
    process_inittopoGMT(inittopo_vec) 

if __name__ == "__main__": main()
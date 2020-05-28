import subprocess
import os
import pandas as pd
import numpy as np

def process_Uplift(ID):

        bashcommand = 'sh file_loop.sh %s' %(ID)

        process = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)

        output, error = process.communicate()

        return

def edit_Uplift(ID,uplift_values):

        uplift = pd.read_csv('AUS/%s/AUS_uplift.csv'%(ID))
        # print(uplift.shape)
        
        uplift.columns = ['0','uplift_name', 'AUS001']

        # uplift.loc[uplift.uplift_name =='Zealandia_Z3_1', 'AUS001'] = 33000
        uplift['AUS001'] = uplift_values
        uplift.to_csv('AUS/%s/AUS_uplift.csv'%(ID), index=False)
        return 

def main():
        ID = 2
        dummy_file = pd.read_csv('AUS/upliftvariables.txt')
        print('dummy_file.shape', dummy_file.shape)
        edit_Uplift(ID, dummy_file)

if __name__ == "__main__": main()
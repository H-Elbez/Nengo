import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class DataLog:
    """
    Class to collect data from simulation and store it somewhere
    """
    def __init__(self):
        self.sim = None
        self.f = None
        self.ToFile = False
        self.ToHeatMap = False

        now = str(datetime.now().time())
        self.folder = "My Sim "+now
    
    def set(self,sim,path,ToFile= False,ToHeatMap= False):
        """
        Set the simulation argument

        Parameters
        ----------
        sim : Simulator
            Simulator argument
        path : str
            Path where to save the log file
        """
        
        self.sim = sim
        self.f = open(path,"w")
        self.ToFile = ToFile
        self.ToHeatMap = ToHeatMap

    def storeToFile(self,label,data):
        """
        Store the log to the file
        """
        i = 0
        for d in data[0]:
            self.f.write(label+":"+str(i)+":"+str(d)+"\n")
            i = i + 1

    def storeToHeatMap(self,data,t):
       
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        i = 0

        for d in data[0]:

            if not os.path.exists(self.folder+"/"+str(i)):
                os.makedirs(self.folder+"/"+str(i))
        
            plt.matshow(np.reshape(d,(28,28)),cmap="Reds")
            plt.title(str(i)+str(t))
            plt.savefig(self.folder+"/"+str(i)+"/"+str(i)+":"+str(i)+str(t)+".png")
            plt.cla()
            
            i = i+1
        
    def closeLog(self):
        """
        Close log file 
        """
        self.f.close()

    def __call__(self, t):
        if self.sim is not None:
            assert len(self.sim.model.probes) != 0 , "No Probes to store"

            for probe in self.sim.model.probes:
                if len(self.sim._sim_data[probe]) != 0: 
                    self.sim._sim_data[probe] = [self.sim._sim_data[probe][-1]]
                    
                    if self.ToFile:
                        self.storeToFile(str(t)+probe.label,self.sim._sim_data[probe])
                    
                    if self.ToHeatMap:
                        self.storeToHeatMap(self.sim._sim_data[probe],t)
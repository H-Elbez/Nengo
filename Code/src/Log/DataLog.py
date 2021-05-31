from datetime import datetime

class DataLog:
    """
    Class to collect data from simulation and keep only last information to avoid filling the memory
    """
    def __init__(self):
        self.sim = None
        self.backend = ""
        self.simName = ""
        self.Dataset = ""
        self.step_time = None
        self.input_nbr = None
        self.date = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    def set(self,sim,Args):
        """
        Set the simulation argument

        Parameters
        ----------
        sim : Simulation instance            
        Args : Simulation arguments
        """
        
        self.sim = sim
        self.backend = Args["backend"]
        self.Dataset = Args["Dataset"]
        self.Labels = Args["Labels"]
        self.step_time = Args["step_time"]
        self.input_nbr = Args["input_nbr"]
        self.simName = self.sim.model.toplevel.label.replace(" ","-")+"-"+self.date

    # ----------------------------------------

    # Process data step by step and remove eveything except last recorded data

    def __call__(self, t):
        
        if self.sim is not None:
            assert len(self.sim.model.probes) != 0 , "No Probes to store"

            for probe in self.sim.model.probes:
                
                if(self.backend == "Nengo"):
                    if len(self.sim._sim_data[probe]) != 0: 
                        self.sim._sim_data[probe] = [self.sim._sim_data[probe][-1]]
                else:
                    if len(self.sim.model.params[probe]) != 0: 
                        self.sim.model.params[probe] = [self.sim.model.params[probe][-1]]
                
    # ----------------------------------------

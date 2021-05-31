from nengo.connection import LearningRule
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import LearningRuleType
from nengo.builder.learning_rules import get_pre_ens,get_post_ens
from nengo.neurons import AdaptiveLIF
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import (NumberParam,Default)
from nengo.dists import Choice
from nengo.utils.numpy import clip
import numpy as np
import random
import math

def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)

#---------------------------------------------------------------------
# Neuron Model declaration 
#---------------------------------------------------------------------

#create new neuron type LIF 

class LIF(AdaptiveLIF):
    probeable = ('spikes', 'voltage', 'refractory_time','adaptation','inhib') #,'inhib'
    
    def __init__(self, spiking_threshold = 1, inhib=[], **lif_args): # inhib=[],T = 0.0
        super(LIF, self).__init__(**lif_args)
        # neuron args (if you have any new parameters other than gain
        # an bais )
        self.inhib = inhib
        #self.T = T
        self.spiking_threshold=spiking_threshold

        self.T = 0
    @property
    def _argreprs(self):
        args = super(LIF, self)._argreprs
        return args

    # dt : timestamps 
    # J : Input currents associated with each neuron.
    # output : Output activities associated with each neuron.
    def step(self, dt, J, output, voltage, refractory_time, adaptation,inhib):#inhib
        self.T = round(self.T+dt,3)
        
        if(np.max(J) != 0):
            J = np.divide(J,np.max(J)) * 1.5



        if math.isclose(math.fmod(self.T,0.20),0,abs_tol=1e-3):
            self.T = 0 
            voltage[...] = 0
            refractory_time[...] = 0
            inhib[...] = 0
            output[...] = 0
            J[...] = 0


        # tInhibit = 10 MilliSecond
        # AdaptiveThresholdAdd = 0.05  millivolts
        # MembraneRefractoryDuration = = 1 MilliSecond
        #print("J",J,"voltage",voltage,output)

        J = J - adaptation
        # ----------------------------

        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = (voltage > self.spiking_threshold)
        output[:] = spiked_mask * (self.amplitude / dt)
        output[voltage != np.max(voltage)] = 0  
        if(np.sum(output) > 1):
            voltage[voltage != np.max(voltage)] = 0 
            output[voltage != np.max(voltage)] = 0
            spiked_mask[voltage != np.max(voltage)] = 0
            inhib[(voltage != np.max(voltage)) & (inhib == 0)] = 15 #10 | 2
        #print("voltage : ",voltage)
        #voltage[inhib != 0] = 0 
        J[inhib != 0] = 0
        #print("\n",dt,J,inhib)
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - self.spiking_threshold) / (J[spiked_mask] - self.spiking_threshold))
        
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = 0
        voltage[refractory_time > 0] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike
        refractory_time[refractory_time < 0] = 0
        # ----------------------------

        adaptation += (dt / self.tau_n) * (self.inc_n * output - adaptation)

        #AdaptiveLIF.step(self, dt, J, output, voltage, refractory_time, adaptation)
        inhib[inhib != 0] += - 1
        #J[...] = 0
        #output[...] = 0
        

#---------------------------------------------------------------------
#add builder for LIF
#---------------------------------------------------------------------

@Builder.register(LIF)
def build_LIF(model, LIF, neurons):
    
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['pre_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.pre_filtered" % neurons)
    model.sig[neurons]['post_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.post_filtered" % neurons)
    model.sig[neurons]['inhib'] = Signal(
        np.zeros(neurons.size_in), name="%s.inhib" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in),name= "%s.adaptation" % neurons
    )
    # set neuron output for a given input
    model.add_op(SimNeurons(neurons=LIF,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            state={"voltage": model.sig[neurons]['voltage'],
                                    "refractory_time": model.sig[neurons]['refractory_time'],
                                    "adaptation": model.sig[neurons]['adaptation'],
                                    "inhib": model.sig[neurons]['inhib']
                                     }))
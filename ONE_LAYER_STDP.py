import sys
import os
sys.path.append(os.getcwd())

import nengo
import pickle
import numpy as np
from numpy import random
from datetime import datetime
from src.Neuron.LIF import LIF
from src.Log.DataLog import DataLog
from nengo_extras.data import load_mnist
from src.LearningRule.simplified_stdp import STDP
from src.Input.InputData import PresentInputWithPause
from src.Log.Heatmap import HeatMapSave,AllHeatMapSave

#############################
# load the data
#############################

img_rows, img_cols = 28, 28
input_nbr = 1000
Dataset = "Mnist"
(image_train, label_train), (image_test, label_test) = load_mnist()

image_train_filtered = []
label_train_filtered = []

for i in range(0,input_nbr):
        image_train_filtered.append(image_train[i])
        label_train_filtered.append(label_train[i])

print("actual input",len(label_train_filtered))
print(np.bincount(label_train_filtered))

image_train_filtered = np.array(image_train_filtered)
label_train_filtered = np.array(label_train_filtered)

#############################

model = nengo.Network(label="My network",)

#############################
# Model construction
#############################

sim_info = {
"presentation_time" : 0.20,
"pause_time" : 0.15,
"n_in" : 784,
"n_neurons" : 20,
"amplitude" : 1,
"dt" : 0.005
}

learning_args = {
            "learning_rate":1e-3,
            "alf_p":0.09,
            "alf_n":0.06,
            "beta_p":2.5,
            "beta_n":1.5,
            "prune":1,
            "stats":False,
            "reinforce":0,
            "BatchPerPrune":(sim_info["presentation_time"]+sim_info["pause_time"])*500}

neuron_args = {
"spiking_threshold":1,
"tau_ref":0.005,
"inc_n":0.04}

# Log and reduce collected Probes to avoid getting memory issues
full_log = False

if(not full_log):
    log = DataLog()

with model:

    # input layer 
    picture = nengo.Node(PresentInputWithPause(image_train_filtered, sim_info["presentation_time"],sim_info["pause_time"]),label="Mnist")
    input_layer = nengo.Ensemble(
        sim_info["n_in"],
        1,
        label="Input",
        neuron_type=nengo.LIF(amplitude=sim_info["amplitude"]),
        gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0])
        )
    input_conn = nengo.Connection(picture,input_layer.neurons,)

    # weights randomly initiated 
    layer1_weights = np.round(random.random((sim_info["n_neurons"], sim_info["n_in"])),5)
    # define first layer
    layer1 = nengo.Ensemble(
         sim_info["n_neurons"],
         1,
         label="layer1",
         neuron_type=LIF(spiking_threshold=neuron_args["spiking_threshold"],tau_ref=neuron_args["tau_ref"],inc_n=neuron_args["inc_n"]),
         intercepts=nengo.dists.Choice([0]),
         max_rates=nengo.dists.Choice([22,22]),
         encoders=nengo.dists.Choice([[1]]),    
         )

  
    conn1 = nengo.Connection(
            input_layer.neurons,
            layer1.neurons,
            transform=layer1_weights, 
            learning_rule_type=STDP(learning_rate=learning_args["learning_rate"],alf_p=learning_args["alf_p"],alf_n=learning_args["alf_n"],beta_p=learning_args["beta_p"],beta_n=learning_args["beta_n"]))

    #############################
    # setup the probes
    #############################

    layer1_synapses_probe = nengo.Probe(conn1,"weights",label="layer1_synapses")
    layer1_voltage_probe = nengo.Probe(layer1.neurons, "voltage", label="layer1_voltage")
    
    if(not full_log):
        nengo.Node(log)

    #############################

step_time = (sim_info["presentation_time"] + sim_info["pause_time"]) 
Args = {"backend":"Nengo","Dataset":Dataset,"Labels":label_train_filtered,"step_time":step_time,"input_nbr":input_nbr}

print(step_time * label_train_filtered.shape[0])
with nengo.Simulator(model,dt=sim_info["dt"],progress_bar=True) as sim:
    
    if(not full_log):
        log.set(sim,Args)

    sim.run(step_time * label_train_filtered.shape[0])


if(not full_log):
    log.closeLog()

now = str(datetime.now().time())
folder = "My_Sim_"+now

if not os.path.exists(folder):
    os.makedirs(folder)
#save the model
pickle.dump([sim.data[layer1_synapses_probe][-1],sim_info,learning_args,neuron_args], open( folder+"/mnist_params_STDP", "wb" ))

print(np.min(sim.data[layer1_synapses_probe]),np.max(sim.data[layer1_synapses_probe]))
print(sim.data[layer1_synapses_probe].shape)

for i in range(0,(sim_info["n_neurons"])):
    if(full_log):
        AllHeatMapSave(sim,layer1_synapses_probe,folder,sim.data[layer1_synapses_probe].shape[0],i)
    else:
        HeatMapSave(sim,folder,layer1_synapses_probe,sim.data[layer1_synapses_probe].shape[0],i)
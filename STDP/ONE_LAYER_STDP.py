import nengo
#import nengo_dl
#import nengo_ocl
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from core.STDPVEC import STDP
from simplified_stdp import STDP
from core.STDP import STDPLIF
from core.DataLog import DataLog
import tensorflow as tf
from core.InputData import PresentInputWithPause
import os
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
import pickle
plt.rcParams.update({'figure.max_open_warning': 0})

#############################
# load the data
#############################

img_rows, img_cols = 28, 28
input_nbr = 1000

(image_train, label_train), (image_test, label_test) = load_mnist()

#select the 0s and 1s as the two classes from MNIST data
image_train_filtered = []
label_train_filtered = []


x = 0
for i in range(0,input_nbr):
#  if label_train[i] == x:
        image_train_filtered.append(image_train[i])
        label_train_filtered.append(label_train[i])
#        if x == 0:
#            x = 1
#        else:
#            x = 0

print("actual input",len(label_train_filtered))
image_train_filtered = np.array(image_train_filtered)
label_train_filtered = np.array(label_train_filtered)

#############################
model = nengo.Network("My network")
#############################

#############################
# Helpfull methodes
#############################

def sparsity_measure(vector):  # Gini index
    # Max sparsity = 1 (single 1 in the vector)
    v = np.sort(np.abs(vector))
    n = v.shape[0]
    k = np.arange(n) + 1
    l1norm = np.sum(v)
    summation = np.sum((v / l1norm) * ((n - k + 0.5) / n))
    return 1 - 2 * summation

def HeatMapSave(network,name,probe,samples,neuron):
    x = int(samples/100)

    if not os.path.exists(name):
        os.makedirs(name)

    plt.matshow(np.reshape(network.data[probe][:,neuron][samples-1],(28,28)))
    plt.title(samples-1)
    plt.savefig(name+"/"+str(neuron)+":"+str(samples-1)+".png")
    plt.cla()

def AllHeatMapSave(network,probe,folder,samples,neuron):

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    os.makedirs(folder+"/"+str(neuron))
    i = 0
    step = samples / 100
    while i < samples:
        plt.matshow(np.reshape(network.data[probe][:,neuron][i],(28,28))) # , cmap=cm.jet
        plt.title(i)
        plt.savefig(folder+"/"+str(neuron)+"/"+str(neuron)+":"+str(i)+".png")
        plt.cla()
        i = int(i + step)

def averageInhibtion(layer,probe):
    # Calculate spike times
    t_spike_diffs = []
    #p1_spike_times = sim.data[p1][:, 0] * sim.trange() * sim.dt
    #p1_spike_times = p1_spike_times[p1_spike_times > 0]

    for nn in range(layer.n_neurons):
        spike_times = sim.data[probe][:, nn] * sim.trange() * sim.dt
        spike_times = spike_times[spike_times > 0]
        for t_spike_p1 in spike_times:
            t_spike_p2 = spike_times[spike_times > t_spike_p1]
            if len(t_spike_p2) > 0:
                t_spike_diffs.append(t_spike_p2[0] - t_spike_p1)
    print("Average spike inhibition time:", np.mean(t_spike_diffs))
#############################

#############################
# Model construction
#############################

presentation_time = 0.35 #0.35
pause_time = 0.0 #0.15
#input layer
n_in = 784
n_neurons = 50

log = DataLog()
with model:
    # Ensemble creation parameters
    e_args = dict(max_rates=nengo.dists.Uniform(22,22),
                  intercepts=nengo.dists.Choice([0]),
                  encoders=nengo.dists.Choice([[1]]),
                  #gain=nengo.dists.Choice([2]),
                  #bias=nengo.dists.Choice([0]),
                  #neuron_type=nengo.neurons.AdaptiveLIF(tau_ref=0.05, amplitude=0.01,inc_n=0.02)
                  )
    # input layer 
    picture = nengo.Node(PresentInputWithPause(image_train_filtered, presentation_time,pause_time))
    input_layer = nengo.Ensemble(
        784,
        1,
        label="input",
        neuron_type=nengo.neurons.SpikingRectifiedLinear(),
        ** e_args
        )
    input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)

    # weights randomly initiated 
    layer1_weights = random.random((n_neurons, 784))

    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,#   noise=nengo.processes.WhiteNoise(),
         label="layer1",
         neuron_type=nengo.AdaptiveLIF(),
         ** e_args
         )

    conn1 = nengo.Connection(
        input_layer.neurons,
        layer1.neurons,
        transform=layer1_weights,
        learning_rule_type=STDP(learning_rate=5e-5,alf_p=0.001,alf_n=0.009,beta_p=1.2,beta_n=0.5)
        )

    # create inhibitory layer 
    inhib_wegihts = (np.full((n_neurons, n_neurons), 1) - np.eye(n_neurons)) * -2

    inhib = nengo.Connection(
        layer1.neurons, 
        layer1.neurons, 
        transform=inhib_wegihts,
        synapse=0.0035
        )
    
    #############################

    #############################
    # setup the probes
    #############################

    connection_layer1_probe = nengo.Probe(conn1,"weights",label="layer1_synapses") # ('output', 'input', 'weights')
    layer1N = nengo.Probe(layer1.neurons)
    inputProbe = nengo.Probe(picture,"output")
    inhibProbe = nengo.Probe(inhib,"output")
    nengo.Node(log)

    #pdm.add_probes([connection_layer1_probe])
    #############################

with nengo.Simulator(model) as sim:

    log.set(sim,"Log.txt",False,False)
    
    sim.run((presentation_time + pause_time) * label_train_filtered.shape[0])

#save the model
pickle.dump(sim.data[connection_layer1_probe][-1], open( "mnist_params_STDP", "wb" ))
log.closeLog()

now = str(datetime.now().time())
folder = "My Sim "+now

#averageInhibtion(layer1,layer1N)

print(np.min(sim.data[connection_layer1_probe]),np.max(sim.data[connection_layer1_probe]))
for i in range(0,(n_neurons)):
    #AllHeatMapSave(sim,connection_layer1_probe,folder,sim.data[connection_layer1_probe].shape[0],i)
    HeatMapSave(sim,folder,connection_layer1_probe,sim.data[connection_layer1_probe].shape[0],i)
#plt.figure()
#plt.subplot(3, 1, 1)
#plt.plot(sim.trange(),sim.data[inputProbe])
#plt.subplot(3, 1, 2)
#plt.plot(sim.trange(),sim.data[layer1N])
#plt.subplot(3, 1, 3)
#plt.plot(sim.trange(),sim.data[inhibProbe])
#plt.show()
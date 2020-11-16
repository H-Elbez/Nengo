import nengo
#import nengo_dl
#import nengo_ocl
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from core.STDP import STDP,STDPLIF
from simplified_stdp import STDP
from core.DataLog import DataLog
import tensorflow as tf
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
input_nbr = 400

(image_train, label_train), (image_test, label_test) = (tf.keras.datasets.mnist.load_data())

#image_train = image_train / 255 
#image_test = image_test / 255

#select the 0s and 1s as the two classes from MNIST data
image_train_filtered = []
label_train_filtered = []

#v = 0
#for i in range(0,input_nbr):
#        image_train_filtered.append(image_train[i])
#        label_train_filtered.append(label_train[i])

#while i < globalInput:
x = 0
for i in range(0,input_nbr):
  if label_train[i] == x:
        image_train_filtered.append(image_train[i])
        label_train_filtered.append(label_train[i])
        if x == 0:
            x = 1
        else:
            x = 0

#for i in range(0,input_nbr):
#   if label_train[i] == 7:
#        image_train_filtered.append(image_train[i])
#        label_train_filtered.append(label_train[i])

#for i in range(0,input_nbr):
#    if label_train[i] == 9:
#       image_train_filtered.append(image_train[i])
#       label_train_filtered.append(label_train[i])

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

def HeatMapSave(network,name,probe,samples=1,neuron=1):
    x = int(samples/100)

    if not os.path.exists(name):
        os.makedirs(name)

    plt.matshow(np.reshape(network.data[probe][:,neuron][samples-1],(28,28)))
    plt.title(samples-1)
    plt.savefig(name+"/"+str(neuron)+":"+str(samples-1)+".png")
    plt.cla()

def AllHeatMapSave(network,probe,folder,samples=1,neuron=1):

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

    #for i in range(0,100):
    #    plt.matshow(np.reshape(network.data[probe][:,neuron][i*x],(28,28)),cmap="Reds")
    #    plt.title(i*x)
    #    plt.savefig(name+"/"+str(neuron)+":"+str(i*x)+".png")
    
#############################

#############################
# Model construction
#############################

presentation_time = 0.20 #0.35
pause_time = 0.10 #0.15
#input layer
n_in = 784
n_neurons = 2

log = DataLog()
with model:

    # input layer 
    picture = nengo.Node(nengo.processes.PresentInput(image_train_filtered, presentation_time))
    input_layer = nengo.Ensemble(
        784,
        1,
        label="input",
        max_rates=nengo.dists.Uniform(22, 22),
        intercepts=nengo.dists.Choice([0])
        )
    input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)

    # weights randomly initiated 
    layer1_weights = random.random((n_neurons, 784))
    #layer1_weights = np.round(layer1_weights,6)
    #layer1_weights = np.full((n_neurons, 784),1)

    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,
         neuron_type=nengo.neurons.AdaptiveLIF(),
         label="layer1",
         max_rates=nengo.dists.Uniform(22,22),
         intercepts=nengo.dists.Choice([0])
         )

    #layer2 = nengo.Ensemble(n_neurons, 1,label="layer2",neuron_type=nengo.neurons.LIFRate(),bias=Choice([0.]),gain=Choice([2.]))
    # define connection between the two layers
    conn1 = nengo.Connection(
        input_layer.neurons,
        layer1.neurons,
        transform=layer1_weights,
        synapse=None,
        learning_rule_type=STDP()
            #learning_rate=1e-5,
            #max_weight=0.99,min_weight=0.01)#,pre_synapse=None learning_rate=1e-4
        )

    # create inhibitory layer 
    inhib_wegihts = (np.full((n_neurons, n_neurons), 1) - np.eye(n_neurons)) * -2

    #for i in range(0,n_neurons):
    #    inhib_wegihts[i,i] = 0

    inhib = nengo.Connection(layer1.neurons, layer1.neurons, transform=inhib_wegihts)
    
    #nengo.Connection(layer2.neurons, layer1.neurons, transform=np.full((n_neurons,n_neurons),-2))
    #nengo.Connection(layer2.neurons,layer1.neurons, transform=np.full((n_neurons,n_neurons),-2))
    
    #############################

    #############################
    # setup the probes
    #############################

   # input_probe = nengo.Probe(input_layer.neurons,"output") # RATELIF : ('output', 'input', 'rates')
    #layer1_probe = nengo.Probe(layer1.neurons,"spikes") # ('output', 'input', 'spikes', 'voltage', 'refractory_time')
    connection_layer1_probe = nengo.Probe(conn1,"weights",label="layer1_synapses") # ('output', 'input', 'weights')
    #connection_layer2_probe = nengo.Probe(conn2,"weights")
    #inhib_input_probe = nengo.Probe(inhib,"input")
    #inhib_output_probe = nengo.Probe(inhib,"output")
    #layer1_voltage_probe = nengo.Probe(layer1.neurons,"voltage")
    #STDP_probe = nengo.Probe(conn1.learning_rule,"delta")

    nengo.Node(log)

    #pdm.add_probes([connection_layer1_probe])
    #############################

with nengo.Simulator(model) as sim:

    log.set(sim,"Log.txt",False,False)
    
    sim.run((presentation_time) * label_train_filtered.shape[0])

#save the model
pickle.dump(sim.data[connection_layer1_probe][-1], open( "mnist_params_STDP", "wb" ))
log.closeLog()

#presentation_time * label_train_filtered.shape[0]
#print(sim.data[connection_layer1_probe].shape[0])
now = str(datetime.now().time())
folder = "My Sim "+now

#print(sim.data[STDP_probe].shape)

#plt.subplot(3, 1, 1)
#plt.plot(sim.trange(),sim.data[STDP_probe][:,0])

#plt.plot(sim.trange(),sim.data[inhib_input_probe][:,0],label="input spikes")
#plt.plot(sim.trange(),sim.data[inhib_output_probe][:,0],label="output spikes")
#plt.plot(sim.trange(),sim.data[conn1].weights[:,0])
#plt.legend()
#plt.grid(True)
#plt.subplot(3, 1, 2) 
#plt.plot(sim.trange(),sim.data[connection_layer1_probe][:,0])
#plt.plot(sim.trange(),sim.data[inhib_output_probe][:,1],label="output spikes")
#plt.legend()
#plt.grid(True)
#plt.subplot(3, 1, 3)
#plt.plot(sim.trange(),sim.data[layer1_voltage_probe][:,0],label="voltage n 1")
#plt.plot(sim.trange(),sim.data[layer1_voltage_probe][:,1],label="voltage n 2")
#plt.legend()
#plt.grid(True)
#plt.show()

#print("time of run : "+str(presentation_time * label_train_filtered.shape[0]))
#print(sim.trange())
print(np.min(sim.data[connection_layer1_probe]),np.max(sim.data[connection_layer1_probe]))
for i in range(0,(n_neurons)):
    #AllHeatMapSave(sim,connection_layer1_probe,folder,sim.data[connection_layer1_probe].shape[0],i)
    HeatMapSave(sim,folder,connection_layer1_probe,sim.data[connection_layer1_probe].shape[0],i)
"""
Evaluation class + using train samples for class attribution
"""


import sys
import os
sys.path.append(os.getcwd())
import nengo
import numpy as np
from nengo_extras.data import load_mnist
import pickle
from src.Log.Stats import classAttribution, evaluation2
from src.Input.InputData import PresentInputWithPause
from src.Neuron.LIF import LIF

#############################
# load the data
#############################

np.random.seed(0)

NetworkInfo = sys.argv[1]

input_nbr = 10000
class_input_nbr = 1000

classes = 10

(image_train, label_train), (image_test, label_test) = load_mnist()

image_test_filtered = []
label_test_filtered = []

class_image_filtered = []
class_label_filtered = []

for i in range(0,input_nbr):
        image_test_filtered.append(image_test[i])
        label_test_filtered.append(label_test[i])

image_test_filtered = np.array(image_test_filtered)
label_test_filtered = np.array(label_test_filtered)

for i in range(0,10):
    indices = np.where(label_train == i)[0]
    p = np.random.permutation(len(indices))
    class_image_filtered.extend(image_train[p][:100])
    class_label_filtered.extend(label_train[p][:100])

p = np.random.permutation(len(class_image_filtered))

class_image_filtered = np.array(class_image_filtered)
class_label_filtered = np.array(class_label_filtered)

class_image_filtered = class_image_filtered[p]
class_label_filtered = class_label_filtered[p]

#############################
# Class attribution
#############################

model = nengo.Network("Class network")

# loaded data
LoadedWeights = pickle.load(open( NetworkInfo, "rb" ))[0]
sim_info = pickle.load(open( NetworkInfo, "rb" ))[1]
learning_args = pickle.load(open( NetworkInfo, "rb" ))[2]
neuron_args = pickle.load(open( NetworkInfo, "rb" ))[3]

with model:

    # input layer 
    picture = nengo.Node(PresentInputWithPause(class_image_filtered, sim_info["presentation_time"],sim_info["pause_time"]))
    input_layer = nengo.Ensemble(
        sim_info["n_in"],
        1,
        label="input",
        neuron_type=nengo.LIF(amplitude=sim_info["amplitude"]),
        encoders=nengo.dists.Choice([[1]]),
        intercepts=nengo.dists.Choice([0]),
        )


    input_conn = nengo.Connection(picture,input_layer.neurons)

    print(sim_info,neuron_args,learning_args)

    try:
        layer1_weights = np.reshape(LoadedWeights,(sim_info["n_neurons"],sim_info["n_in"]))
    except:
        assert False , "Make sure the neurons nbr is the same"


    # define first layer
    layer1 = nengo.Ensemble(
         sim_info["n_neurons"],
         1,
         label="layer1",
         neuron_type=LIF(spiking_threshold=neuron_args["spiking_threshold"],tau_ref=neuron_args["tau_ref"],inc_n=neuron_args["inc_n"],tau_rc=neuron_args["tau_rc"],amplitude=0.001),#
         gain=nengo.dists.Choice([2]),
         encoders=nengo.dists.Choice([[1]]),
         bias=nengo.dists.Choice([0])
         )

    # define connection between the two layers
    conn1 = nengo.Connection(
        input_layer.neurons,
        layer1.neurons,
        transform=layer1_weights)

    #############################
    # setup the probes
    #############################

    class_spikes_layer1_probe = nengo.Probe(layer1.neurons,"spikes") # ('output', 'input', 'spikes', 'voltage', 'refractory_time')

simTime = (sim_info["presentation_time"]+sim_info["pause_time"])*class_label_filtered.shape[0]

with nengo.Simulator(model,dt=sim_info["dt"]) as sim:

    sim.run(simTime)

neuronClasses = classAttribution(classes,sim_info["n_neurons"],int((simTime / sim.dt) / class_input_nbr) ,sim.data[class_spikes_layer1_probe],class_label_filtered)

print("Classes :\n",neuronClasses,"\n")
#############################
# Model construction
#############################

model = nengo.Network("My network")

# loaded data
LoadedWeights = pickle.load(open( NetworkInfo, "rb" ))[0]
sim_info = pickle.load(open( NetworkInfo, "rb" ))[1]
learning_args = pickle.load(open( NetworkInfo, "rb" ))[2]
neuron_args = pickle.load(open( NetworkInfo, "rb" ))[3]

with model:

    # input layer 
    picture = nengo.Node(PresentInputWithPause(image_test_filtered, sim_info["presentation_time"],sim_info["pause_time"]))
    input_layer = nengo.Ensemble(
        sim_info["n_in"],
        1,
        label="input",
        neuron_type=nengo.LIF(amplitude=sim_info["amplitude"]),
        intercepts=nengo.dists.Choice([0]),
        max_rates=nengo.dists.Choice([22,22]),
        encoders=nengo.dists.Choice([[1]]),
        )

    input_conn = nengo.Connection(picture,input_layer.neurons)

    try:
        layer1_weights = np.reshape(LoadedWeights,(sim_info["n_neurons"],sim_info["n_in"]))
    except:
        assert False , "Make sure the neurons nbr is the same"


    # define first layer
    layer1 = nengo.Ensemble(
         sim_info["n_neurons"],
         1,
         label="layer1",
         neuron_type=LIF(spiking_threshold=neuron_args["spiking_threshold"],tau_ref=neuron_args["tau_ref"],inc_n=neuron_args["inc_n"]),#
         gain=nengo.dists.Choice([2]),
         encoders=nengo.dists.Choice([[1]]),
         bias=nengo.dists.Choice([0])
         )

    # define connection between the two layers
    conn1 = nengo.Connection(
        input_layer.neurons,
        layer1.neurons,
        transform=layer1_weights)

    #############################
    # setup the probes
    #############################

    spikes_layer1_probe = nengo.Probe(layer1.neurons,"spikes") # ('output', 'input', 'spikes', 'voltage', 'refractory_time')

simTime = (sim_info["presentation_time"]+sim_info["pause_time"])*label_test_filtered.shape[0]

with nengo.Simulator(model,dt=sim_info["dt"]) as sim:

    sim.run(simTime)

print("Accuracy :",evaluation2(int((simTime / sim.dt) / input_nbr) ,sim.data[spikes_layer1_probe],label_test_filtered,neuronClasses),"%")
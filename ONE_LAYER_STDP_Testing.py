import nengo
import numpy as np
from nengo_extras.data import load_mnist
from nengo.dists import Choice
import pickle
from core.STDP import STDPLIF
from simplified_stdp import STDP
#############################
# load the data
#############################

img_rows, img_cols = 28, 28
input_nbr = 2000
classes = 10

(image_train, label_train), (image_test, label_test) = load_mnist()

image_test_filtered = []
label_test_filtered = []

for i in range(0,input_nbr):
#   if label_test[i] == 7:
        image_test_filtered.append(image_test[i])
        label_test_filtered.append(label_test[i])

#for i in range(0,input_nbr):
#    if label_test[i] == 9:
#       image_test_filtered.append(image_test[i])
#       label_test_filtered.append(label_test[i])

image_test_filtered = np.array(image_test_filtered)
label_test_filtered = np.array(label_test_filtered)

#############################
model = nengo.Network("My network")
#############################

#############################
# Helpfull methodes
#############################

def evaluation(classes,n_neurons,presentation_time,spikes_layer1_probe):

    detection = np.zeros((classes,n_neurons))
    
    x = 0
   # print(sim.data[spikes_layer1_probe][x*presentation_time:(x+1)*presentation_time])
    for i in label_test_filtered:
         tmp = sim.data[spikes_layer1_probe][x*presentation_time:(x+1)*presentation_time].sum(axis=0)

         detection[i] = detection[i] + (tmp / (1/ sim.dt))

         x = x + 1
    
    Good = Bad = 0

    for i in range(0,n_neurons):
        Good = Good + np.max(detection[:,i])
        Bad = Bad + (np.sum(detection[:,i]) - np.max(detection[:,i]))
    
    print("Accuracy :",round((Good * 100)/(Good + Bad),2),"%")

#############################
# Model construction
#############################

presentation_time = 0.35 #0.35
pause_time = 0.15 #0.15
#input layer
n_in = 784
n_neurons = 20

with model:

    e_args = dict(max_rates=nengo.dists.Uniform(22,22),
                intercepts=nengo.dists.Choice([0]),
                encoders=nengo.dists.Choice([[1]]))
                
    # input layer 
    picture = nengo.Node(nengo.processes.PresentInput(image_test_filtered, presentation_time))
    input_layer = nengo.Ensemble(
        784,
        1,
        label="input",
        neuron_type=nengo.neurons.SpikingRectifiedLinear(),
        ** e_args
        )
    input_conn = nengo.Connection(picture,input_layer.neurons)

    # weights loaded initiated
    LoadedWeights = pickle.load(open( "mnist_params_STDP", "rb" )) 
    layer1_weights = np.reshape(LoadedWeights,(n_neurons,784))

    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,
         neuron_type=nengo.AdaptiveLIF(),
         label="layer1",
         ** e_args
         )

    # define connection between the two layers
    conn1 = nengo.Connection(
        input_layer.neurons,
        layer1.neurons,
        #learning_rule_type=STDP(learning_rate=5e-5,alf_p=0.001,alf_n=0.009,beta_p=1.2,beta_n=0.5),
        transform=layer1_weights)

    # create inhibitory layer 
    inhib_wegihts = (np.full((n_neurons, n_neurons), 1) - np.eye(n_neurons)) * -2

    inhib = nengo.Connection(layer1.neurons, layer1.neurons, transform=inhib_wegihts)

    #############################

    #############################
    # setup the probes
    #############################
  
    spikes_layer1_probe = nengo.Probe(layer1.neurons,"spikes") # ('output', 'input', 'spikes', 'voltage', 'refractory_time')

simTime = presentation_time*label_test_filtered.shape[0]

with nengo.Simulator(model) as sim:

    sim.run(simTime)

evaluation(classes,n_neurons,int((simTime / sim.dt) / input_nbr) ,spikes_layer1_probe)
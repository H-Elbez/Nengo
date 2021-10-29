import numpy as np 

""" 

how to call:
accuracy = evaluation(classes,n_neurons,int((simTime / sim.dt) / input_nbr) ,sim.data[spikes_layer1_probe],label_test_filtered,sim.dt)

classes = 10
n_neurons = 20
simTime = (presentation_time+pause_time)*label_test_filtered.shape[0]
input_nbr = 10000
...
..
.
"""
def evaluation(classes,n_neurons,presentation_time,spikes_layer1_probe,label_test_filtered,dt):
    
    ConfMatrix = np.zeros((classes,n_neurons))
    labels = np.zeros(n_neurons)
    accuracy = np.zeros(n_neurons)
    total = 0
    Good = 0
    Bad = 0
    # confusion matrix
    x = 0
    for i in label_test_filtered:
            tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
            tmp[tmp < np.max(tmp)] = 0
            tmp[tmp != 0] = 1
            
            ConfMatrix[i] = ConfMatrix[i] + tmp

            x = x + 1
            

    Classes = dict()
    for i in range(0,n_neurons):
        Classes[i] = np.argmax(ConfMatrix[:,i])
    
    x = 0
    for i in label_test_filtered:
        correct = False
        tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
        tmp[tmp < np.max(tmp)] = 0
        tmp[tmp != 0] = 1

        for index,l in enumerate(tmp):
            if(l == 1):
                correct = correct or (Classes[index] == i)
        if(correct):
            Good += 1
        else:
            Bad += 1
        x = x + 1
        total += 1

    return round((Good * 100)/(Good+Bad),2)

def classAttribution(classes,n_neurons,presentation_time,spikes_layer1_probe,label_test_filtered):
    
    ConfMatrix = np.zeros((classes,n_neurons))
    # confusion matrix
    x = 0
    for i in label_test_filtered:
            tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
            tmp[tmp < np.max(tmp)] = 0
            tmp[tmp != 0] = 1
            ConfMatrix[i] = ConfMatrix[i] + tmp
            x = x + 1
            
    Classes = dict()
    for i in range(0,n_neurons):
        Classes[i] = np.argmax(ConfMatrix[:,i])
    
    return Classes

def evaluation2(presentation_time,spikes_layer1_probe,label_test_filtered,Classes):
    
    total = 0
    Good = 0
    Bad = 0
    
    x = 0
    for i in label_test_filtered:
        correct = False
        tmp = spikes_layer1_probe[(x*presentation_time):(x+1)*presentation_time].sum(axis=0)
        tmp[tmp < np.max(tmp)] = 0
        tmp[tmp != 0] = 1

        for index,l in enumerate(tmp):
            if(l == 1):
                correct = correct or (Classes[index] == i)
        if(correct):
            Good += 1
        else:
            Bad += 1
        x = x + 1
        total += 1

    return round((Good * 100)/(Good+Bad),2)
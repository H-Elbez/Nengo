import os 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

def HeatMapSave(network,name,probe,samples,neuron):
    x = int(samples/100)

    if not os.path.exists(name):
        os.makedirs(name)

    plt.imshow(np.reshape(network.data[probe][:,neuron][samples-1],(28,28)),interpolation='none',cmap="jet",vmin=0,vmax=1)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(name+"/"+str(neuron)+":"+str(samples-1)+".png", pad_inches = 0,bbox_inches='tight')
    plt.cla()

def AllHeatMapSave(network,probe,folder,samples,neuron):

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    os.makedirs(folder+"/"+str(neuron))
    i = 0
    step = samples / 100
    print("Saving Heatmaps ...")
    while i < samples:
        plt.imshow(np.reshape(network.data[probe][:,neuron][i],(28,28)),interpolation='none',cmap="jet",vmin=0,vmax=1) # , cmap=cm.jet
        plt.savefig(folder+"/"+str(neuron)+"/"+str(i).zfill(10)+".png", pad_inches = 0,bbox_inches='tight',transparent=True)
        plt.cla()
        plt.close()
        i = int(i + step)
    
    print("Generate Video from Heatmaps ...")
    os.system("ffmpeg -pattern_type glob -i '"+os.getcwd()+"/"+folder+"/"+str(neuron)+"/"+"*.png' -vcodec mpeg4 -hide_banner -loglevel panic -y "+os.getcwd()+"/"+folder+"/"+str(neuron)+".mp4")
    os.system("rm "+os.getcwd()+"/"+folder+"/"+str(neuron)+" -R")
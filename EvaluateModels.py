import sys

#ys.path.append("/home/chlw/software/repositories/fire-forecast")

from fire_forecast.deep_learning.models import load_model_from_config
from fire_forecast.deep_learning.iterator import Iterator
from fire_forecast.deep_learning.utils import read_config
from fire_forecast.deep_learning.ensemble import Ensemble
import yaml
import torch
from fire_forecast.deep_learning.utils import flatten_features, flatten_labels_and_weights
import matplotlib.pyplot as plt
import numpy as np
from fire_forecast.evaluation import evaluation
import argparse


# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script evaluate models and plot example prediction, run with: python EvaluateModels.py /data/runs/final0_2day_meteo/config.yaml -Plot_TimeSeries 550 -savepath /home/username/Plots/")
    parser.add_argument("config", nargs = "+", help="path to NN model config file, config file needs to have checkpoint file included, if multiple configs provided the Ensemble evaluated")
    parser.add_argument("-Plot_TimeSeries", type=int, help="optional, to plot an example timeseries enter a number, will be used as index to select a timeseries")    
    parser.add_argument("-savepath", type=str, help="folderpath to save plots")
    args = parser.parse_args()
    return args

# SETTINGS:
args = parse_arguments()
configPath = args.config
print(configPath)
if args.Plot_TimeSeries is not None:
    nTimeseries = args.Plot_TimeSeries
    if args.savepath is None:
        print('Outputfigures will be saved in current folder, please provide a folderpath to save the figures if wanted otherwise')
        savepath = '.'
    else:
        savepath = args.savepath

config = read_config(configPath[0])
iterator = Iterator(config)


# Training the classic models

fire_features, meteo_features, labels = iterator.train_dataset[:]
features = flatten_features(fire_features, meteo_features)
target_values, weights = flatten_labels_and_weights(labels)

# The training features with shape (n_samples, n_features)
X: np.array = features
# The training targets with shape (n_samples, n_targets)
y: np.array = target_values

models = evaluation.train_models(X, y)


# Testing 

fire_features, meteo_features, labels = iterator.test_dataset[0:10000]
features = flatten_features(fire_features, meteo_features)
target_values, weights = flatten_labels_and_weights(labels)

# Test the NN 
if len(configPath) > 1:
    configs = [read_config(config)["model"] for config in configPath]
    ensemble = Ensemble(*configs) #Load the ensemble of models
    ensemble_mean, ensemble_std = ensemble.predict(fire_features, meteo_features) #Predict with the ensemble
    print(ensemble_mean.shape)
    predictions = ensemble_mean#.numpy()
else:
    model = iterator.model
    with torch.no_grad():
        predictions = model(torch.from_numpy(features))
        predictions = predictions.numpy()

# test the classic models
# The test features with shape (n_samples, n_features)
X_test: np.array = features
# The test targets with shape (n_samples, n_targets)
y_test: np.array = target_values

# The predictions of other models as a dictionary of model name to y_pred with shape (n_samples, n_targets)
predictions_dict: dict = {
    "NN": predictions,
    "Persistence": fire_features[:,0,:,0,0],
}
# The weights of the labels as np.array with shape (n_samples, n_targets)
weights_test: np.array = weights

#Calculate metrics
metrics = evaluation.evaluate_models(
    models=models,
    X=X_test,
    y_true=y_test,
    predictions=predictions_dict,
    weights=weights_test,
)

print(metrics.head())

# plot example figures
if args.Plot_TimeSeries is not None:
    y_pred = evaluation.predict(models, X_test)

    # plot predictions of classic models, NN and persistence
    colorlist = ['red','blue','green','orange','purple']
    ncolor = 0
    #figure without meteo and all models
    fig, ax3 = plt.subplots()
    ax3.plot(range(25,49),labels[nTimeseries,0],color = 'black',marker = '',label = 'measurement')
    for name, pred in predictions_dict.items():
        ax3.plot(range(25,49),pred[nTimeseries],color = colorlist[ncolor],marker = '',label = 'prediction '+name)
        ncolor = ncolor + 1
    for name in models.keys():
        ax3.plot(range(25,49),y_pred[name][nTimeseries],color = colorlist[ncolor],marker = '',label = 'prediction '+name)
        ncolor = ncolor + 1
    ax3.plot(range(1,25),fire_features[nTimeseries,0,:,0,0],color = 'black',marker = '')

    ax3.pcolorfast((24.5,48.5), ax3.get_ylim(),
            labels[nTimeseries,1][np.newaxis],
            cmap='binary', alpha=0.3)
    ax3.pcolorfast((0.5,24.5), ax3.get_ylim(),
            fire_features[nTimeseries,1,:,0,0][np.newaxis],
            cmap='binary', alpha=0.3,label = 'offire')
    ax3.set_ylabel('frp')
    ax3.set_xlabel('time step [h]')
    ax3.legend(loc = 2)
    plt.savefig(savepath+'/predictionFRP_bg48_final0_day_meteo_'+str(nTimeseries)+'.png',bbox_inches='tight')

    #Figure with NN prediction and meteo data
    fig, ax2 = plt.subplots()
    secAx = ax2.twinx()
    secAx2 = ax2.twinx()
    ax2.plot(range(25,49),labels[nTimeseries,0],color = 'black',marker = '',label = 'measurement')
    ax2.plot(range(25,49),predictions[nTimeseries],color = 'red',marker = '',label = 'prediction')
    if len(configPath) > 1:
        ax2.fill_between(range(25,49),predictions[nTimeseries]-ensemble_std[nTimeseries],predictions[nTimeseries]+ensemble_std[nTimeseries],color = 'red',alpha = 0.3 )
    ax2.plot(range(1,25),fire_features[nTimeseries,0,:,0,0],color = 'black',marker = '')
                                                                                                                
    ax2.pcolorfast((24.5,48.5), ax2.get_ylim(),
        labels[nTimeseries,1][np.newaxis],
        cmap='binary', alpha=0.3)
                                                                                            
    ax2.pcolorfast((0.5,24.5), ax2.get_ylim(),
        fire_features[nTimeseries,1,:,0,0][np.newaxis],
        cmap='binary', alpha=0.3,label = 'offire')
    print(meteo_features.shape)
    secAx.plot(range(1,49),meteo_features[nTimeseries,0,:,0,0],ls = ':',color = 'darkgreen',marker = '',label = 'VSWL')
    secAx2.plot(range(1,49),meteo_features[nTimeseries,1,:,0,0],ls = ':',color = 'blue',marker = '',label = 'Temp.')
    ax2.set_ylabel('frp')
    ax2.set_xlabel('time step [h]')
    ax2.legend(loc = 2)
    secAx.legend(loc = 1)
    secAx2.legend(loc = 9)
    secAx.set_ylabel('Vol. soil water layer 1')
    secAx2.set_ylabel('Skin Temperature')
    #secAx2.set_ylabel(r'$\rm Fire~emissions~[Tg~CO_2]$',fontsize=20)
    secAx2.spines["right"].set_position(("axes", 1.2))
    secAx2.set_frame_on(True)
    secAx2.patch.set_visible(False)
    for sp in secAx2.spines.values():
        sp.set_visible(False)
    secAx2.spines["right"].set_visible(True)
    plt.savefig(savepath+'/predictionFRP_bg_wMeteo_final0_day_meteo_0_1_'+str(nTimeseries)+'.png',bbox_inches='tight')


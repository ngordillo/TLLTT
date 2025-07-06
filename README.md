# *This* Looks Like *That There* - Application of an Interpretable Prototypical-Part Network to Subseasonal-to-Seasonal Climate Prediction over North America
***


In recent years, the use of neural networks for weather and climate prediction has greatly increased. In order to explain the decision-making process of machine learning “black-box” models, recent research has focused on machine learning explainability methods (XAI). An alternative approach is to build inherently interpretable neural network architectures that can be understood by a human throughout the entire decision-making process, rather than explained post-hoc. Here, we apply such a neural network architecture, named ProtoLNet, in a subseasonal-to-seasonal climate prediction setting. ProtoLNet identifies predictive patterns in the training data that can be used as prototypes to classify the input, while also accounting for the absolute location of the prototype in the input field. In our application, we use data from the Community Earth System Model version 2 (CESM2) pre-industrial long control simulation and train ProtoLNet to identify prototypes in precipitation anomalies over the Indian and North Pacific Oceans to forecast 2-meter temperature anomalies across the western coast of North America on subseasonal-to-seasonal timescales. These identified CESM2 prototypes are then projected onto fifth-generation ECMWF Reanalysis (ERA5) data to predict temperature anomalies in the reanalysis several weeks ahead. We compare the performance of ProtoLNet between using CESM2 and ERA5 data and then demonstrate a novel transfer learning approach which allows us to identify skillful prototypes in the reanalysis. We show that the predictions by ProtoLNet using both datasets have skill while also being interpretable, sensible, and useful for drawing conclusions about what the model has learned.

## Tensorflow Code
***
This code was written in python 3.9.4 with tensorflow 2.5.0 and numpy 1.20.1. 

Within the ProtoLNet code:
* ```experiment_settings.py``` specificies the experiment parameters throughout the code
* ```_pretrain_CNN.ipynb``` pre-trains the base CNN if desired
* ```_main_CNN.ipynb``` trains the ProtoLNet
* ```_vizPrototypes.ipynb``` computes the prototypes and displays them

### Python Environment
The following python environment was used to implement this code.
```
- conda create --name env-tf2.5-cartopy
- conda activate env-tf2.5-cartopy
- conda install anaconda
- pip install tensorflow==2.5 silence-tensorflow memory_profiler  
- conda install -c conda-forge cartopy
- pip uninstall shapely
- pip install --no-binary :all: shapely
- conda install -c conda-forge matplotlib cmocean xarray netCDF4 
- conda install -c conda-forge cmasher cmocean icecream palettable seaborn
- pip install keras-tuner --upgrade
```

#### Funding sources
This work was funded, in part, by the NSF AI Institute for Research on Trustworthy AI in Weather, Climate, and Coastal Oceanography ([AI2ES](https://www.ai2es.org/)).

### References for the *"TApplication of an Interpretable Prototypical-Part Network to Subseasonal-to-Seasonal Climate Prediction over North America"* 
* In prep at AIES

### Fundamental references for this work

### License
This project is licensed under an MIT license.







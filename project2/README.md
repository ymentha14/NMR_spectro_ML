## A machine learning approach to determine chemical shifts in NMR spectroscopy data 
_____________________________________________________________________
**Machine Learning (CS-433)** - Project 2, December 2019

_Yann Mentha, Maxime Epars, Gianni Giusto_

### Dataset
The  dataset is divided into a training and a testing set composed of 250’000 and 568’238 samples respectively and both having 30 features. The training set is paired with labels where each sample is associated  to  a  category  (−1 for  background  noise  and 1 for the presence of a Higgs Boson).

### Code architecture
The code is separated into 2 distinct files:

>1. run.py
>2. implementations.py

The `run.py` file contains the main and can be run in a terminal. 

The `implementations.py` file contains all useful functions and is divided into 5 sections: 

   - "IMPLEMENTATIONS" contains the 6 working methods from the labs, that is: `least_squares_GD`, `least_squares_SGD`, `least_squares`,        `ridge_regression`, `logistic_regression` and `reg_logistic_regression`.
   - "UTILITARIES" contains functions that are called by the 6 methods from "IMPLEMENTATIONS" (e.g. gradient computation) and other             various handy methods 
   - "DATA PROCESSING" contains all the functions that are used to process the data, perform feature engineering, ...
   - "DATA VISUALIZATION" contains all the functions that are made to display the results in figures.
   - "HELPERS" contains the function provided by the teaching team. They allow to load the data, predict the labels and create a   
      submission file in `.csv` format. 
    

### Data Pipeline
![Data processing pipeline](https://github.com/ymentha14/MLprojectfall2019/blob/master/project1/results/pipeline.png)

### Code execution
Run the following command line in the terminal : `python3 run.py`


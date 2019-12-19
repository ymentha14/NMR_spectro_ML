## A machine learning approach to determine chemical shifts in NMR spectroscopy data 
_____________________________________________________________________
**Machine Learning (CS-433)** - Project 2, December 2019

_Maxime Epars, Gianni Giusto, Yann Mentha_


### About the project
For this _Machine Learning for science_ project, we worked together with the Laboratory of Computational Science and Modeling (COSMO) at EPFL and aimed to determine the chemical shifts of atoms from solid-state nuclear magnetic resonance (NMR) spectroscopy data.

### Dataset
Features describe the local environment of each atom as a neighborhood density, represented by a superposition of gaussians centered on each of the atom positions contained in the 3D space.  We were provided with 3 datasets each consisting of a different local environment (_i.e._ corresponding to a different cut-off radius) of 3, 5 and 7 Angstrom.

### Running instructions
To properly run the current project, follow carefully the following steps.

```
Step 1
Download the datafiles shared by the COSMO lab: https://drive.google.com/drive/u/1/folders/1Iz_aDqnylGmsQHzHSxSlvsPRmFhwVcoM
```

```
Step 2
Download the code (*.zip* format) from the course submission plateform: XXX
```

```
Step 3
From the zip archive, install the required packages listed in the .yml file.
The easiest solution is to create an environment with the required packages using: conda env create -f environment.yml
```

```
Step 4
Open either the project2_ann_mac.ipynb or data_story.ipynb notebooks and run all the kernel to have a closer look at the job achieved throughout this project.
```


### Code architecture
The code is separated into distinct files:

>1. data_story.ipynb
>2. project2_ann_mac.ipynb
>3. helpers.py
>4. pickle file

The `data_story.ipynb` file aims at displaying striking figures and scores and hence summarize the work achieved on the current dataset. Therefore, only best methods are used. For complementary information about the different pipeline tested you can refer to the `project2_ann_mac.py` file.

The `project2_ann_mac.py` file contains the whole code subdivided into section following our pipeline of data processing.

The `helpers.py` file contains all useful function required to load and preprocess the data but also to train models and estimate performances.

The `pickle files` contains all pre-runned test to save time when displaying the results.

   - "IMPLEMENTATIONS" contains the 6 working methods from the labs, that is: `least_squares_GD`, `least_squares_SGD`, `least_squares`,        `ridge_regression`, `logistic_regression` and `reg_logistic_regression`.
   - "UTILITARIES" contains functions that are called by the 6 methods from "IMPLEMENTATIONS" (e.g. gradient computation) and other             various handy methods 
   - "DATA PROCESSING" contains all the functions that are used to process the data, perform feature engineering, ...
   - "DATA VISUALIZATION" contains all the functions that are made to display the results in figures.
   - "HELPERS" contains the function provided by the teaching team. They allow to load the data, predict the labels and create a   
      submission file in `.csv` format. 
    

### Data processing pipeline
![Data processing pipeline](https://github.com/ymentha14/MLprojectfall2019/blob/master/project1/results/pipeline.png)




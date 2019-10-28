# Machine Learning (CS-433)
_____________________________________________________________________
Project 1, October 2019

_Yann Mentha, Maxime Epars, Gianni Giusto_


### Dataset
The  dataset is divided into a training and a testing set composed of 250’000 and 568’238 samples respectively and both having 30 features. The training set is paired with labels where each sample is associated  to  a  category  (−1 for  background  noise  and 1 for the presence of a Higgs Boson).

### Code architecture
The code is separated into 3 distinct files:

>1. run.py
>2. implementations.py
>3. proj1_helpers.py

The `run.py` file contains the main and can be run in a terminal. 

The `implementations.py` file contains all useful functions and is divided into 3 sections: 

   - "Implementations" contains the 6 working methods from the labs, that is: `least_squares_GD`, `least_squares_SGD`, `least_squares`,        `ridge_regression`, `logistic_regression` and `reg_logistic_regression`.
   - "Utilitaries" contains all the methods that are needed in exploratory data analysis and feature engineering as well as all the              methods that are used by the 6 methods from "Implementations".
   - "Visualization" contains all the functions that are made to display the results in figures.
    
The `proj1_helpers.py`file contains the functions provided by the teaching team. They allow to load the data, predict the labels and create a submission file in `.csv` format. 



### Code execution

### References

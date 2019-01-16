# ELDTM_ENSAE2018
(Python) subject


### Projet Eléments logiciels pour le traitement des données massives de: <br/> Pierre Le Pelletier de Woillemont  & Cédric Vincent-Cuaz

#### This repository is made as such :

  1. _train-images-idx3-ubyte_ : the MNSIT pictures dataset (60k images of dimensions (28,28)) 
  2. _train-labels-idx1-ubyte_ : the labels corresponding to the MNSIT 
  3. _CNN_ : the folder containing the numpy implementation 
  4. _Nb cnn_ : the folder containing the Numba implementation 
  5. **Rendu_ELTDM.ipynb** : This is the main file

#### Each of the folder is basically made of these scripts:

  1. _forward.py_ : all the necessary forward functions (convolution, maxpool, softmax, categorical cross entropy, dense layer) 
  2. _backward.py_: all the necessary backward functions (same functions as in forward, but for the backward passes) 
  3. _network.py_ : this is where we define our model, by calling the function from the 2 previous files. And also define the learning structure (with batches definition and so on)
  4. _utils.py_
 
 
 It is advised to download this repo and open the notebook __Rendu_ELTDM.ipynb__ on jupyter, otherwise just looking at it on the Github may cause some display problems.

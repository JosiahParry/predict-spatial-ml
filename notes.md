Currently, explicitly spatial models make the assumption that the spatial features are fixed. Meaning that we training and testing data come from the same adjacency matrix. 

Or, we assume that the models are inductive.

Transductive: use the fixed graph always
Inductive: learn a pattern from the model and apply it to an entirely new graph


When we train a model on a fixed set of locations, this is fine so long as we are making predictions to all of the same locations at the same time. If we have $W_x$ (spatial lag of x) as a covariate, we require that all values of $x$ exist so that we can perform the spatial lag. We cannot perform a prediction on a subset of the locations without having the full vector of $x$ 

^ - is that a transductive problem?

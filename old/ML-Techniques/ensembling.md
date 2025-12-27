# Ensembling

## Overview
***
### Section Sources
* *Designing Machine Learning Systems*, Huyen 2022. Ch 6
***

### Bagging
* Aka 'boostrap aggregating'
* Reduces variance & helps avoid overfitting
* Sample from dataset with replacement to create different datasets, called bootstraps
* Train a model on each boostrap
* Final prediction is decided by majority vote of the models (if classification, otherwise avg of all)
* Sampling with replacement ensures independence of samples 
* Generally improves unstable methods like NN's, decision & regression trees, but can degrade performance of stable methods like K-NN
* Random Forest is bagging added to decision/regression tree (along with feature randomness)

### Boosting
* A family of iterative algos that convert weak learners to strong ones
* Each learner is trained on the same set of samples, but samples are weighted differently in each iteration
* Therefore each iteration learner focuses on examples previous weak learner's misclassified
1. Train first weak classifier on original dataset
2. Re-weight samples based on how well the first classifier classified them
3. Train the second classifier on the re-weighted dataset 
4. Samples are weighted based on how well the existing ensemble (now two weak classifiers) weights them
5. Third weak classifier is trained on this re-weighted classifier then added to the ensemble.
6. Repeat
7. Final classifier is a weighted combo of all weak classifiers. Classifiers w/smaller training errors have larger weights
Use LightGBM for distributed gradient boosting w/parallel learning

### Stacking
Train base learners from training data, then a final model on their output (called a meta-learner) for the final prediction. The meta-learner can be a simple heuristic or another model

***





# Experiment Tracking

## *Designing Machine Learning Systems*, Huyen 2022. Ch 6
***

* Model versioning (saving all the details of an expirment to re-create it later) goes hand and hand with experiment tracking

### During Training
* During training, experiment tracking can perform "Babysitting" of training process
* Loss curve - loss on train split and eval split per epoch
* training speed(number of steps /sec or tokens/sec)
* system performance - memory & CPU/GPU %
* model performance - F1, accuracy ,etc
* parameter and hyperparameter values that change over time like learning rate (if using learning rate schedule), gradient norms (globally and per layer) and weight norms

### Versioning
* Models are part data, part code
* 

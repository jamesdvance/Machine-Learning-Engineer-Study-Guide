# Designing Machine Learning Systems
### Chip Huyen 2022

## Overview of Machine Learning Systems

## Introduction of Machine Learning Systems Design

## Data Engineering Fundementals

### Sampling

#### Non Probability Sampling
Sampling method where sampling isn't based on probability criteria. It's based on things like availability, etc
* Convience sampling - based on whatever is available 
* Snowball sampling - samples are based on existing samples (e.g. recursively scraping all users followed on Twitter)
* Judgement sampling - experts decide
* Quota sampling - select slices of data without randomization based on attribute quotes

#### Non-Prob examples
* LMs trained on Common Crawl, Reddit
* Self-driving cars trained on sunny weather b/c programs in AZ and CA

#### Random Sampling
Randomly selected a given percentage. Is susceptible to leaving out rare examples

#### Stratified Sampling
1. Divide population into groups desired to have representation in sample
2. Randomly sample from each group

Is complicated by multiple labels and too many groups. 

#### Weighted Sampling
Samples are selected with a probability according to some weight they're given
* Good when you want to oversample more recent data
* Also when your data has a different distribution than the ground truth

#### Reservoir Sampling
Especially useful when dealing with streaming data. If you always want to sample k, but there's no way of knowing the true sample size ahead of time. Reservior sampling helps so that the sampling can be stopped any time samples maintain a consistent probability. 

1. Put first k elements into the reservoir
2. For each incoming nth element, generate a random number i where 1 <= i <=n. 
3. If 1 <= i <= k, replace the ith element in the reservoir with the nth element

This keeps each incoming element's probability of being in the sample as k/n. This means all samples have an equal probablity of being selected, and can stop any time. 

#### Importance Sampling
Importance sampling allows one to sample from a distribution when only having access to another distribution. It selects samples from the available distribution based on their proportion to the desired distribution.

* Sample is weighted by P(x)/Q(x) as you select from Q(x)
* Q(x) is called the proposal distribution (or importance distribution)
* P(x) is the desired distribution

One use case is policy-based reinforcement learning. Calculating total rewards of the new policy is too costly since it includes all outcomes until the end of the time horizon. But if the new policy is relatively close to the old policy, can sample from the old policy, reweighted so they reflect the new policy. 

### Labeling

#### Hand Labels
* Hand labels are expensive and scale linearly
* The more specialized knowledge is required for labelng, the more likely two labelers could disagree
* Label multiplicity can occur without strict label rules (multiple labels on one sample)
* Data lineage is crucial to be able to make changes to batches of data based on the quality of their labels

#### Natural Labels
* Recommender systems are the classic example of natural labels
* Other examples are stock prices, trip time, ads
* A click is an instant positive label. Some period (say 10 mins) without a click can make an implicit negative label
* Sometimes natural labels can be added as part of an app, for example prompting the user for a better translation of language
* Like and reaction buttons in social media create natural labels which otherwise wouldnt' exist
* Length of feedback loop is important to consider. Some users take up recommendations after a period, especially for longer content like movies or books
* There's a trade-off between speed and accuracy in setting a feedback loop time limit

#### Handling Lack Of Labels
* Weak supervision
* Semi-supervision
* Transfer learning
* Active learning

##### Weak supervision
* Heuristics used to generate labels. 
* Aka programatic labeling. 
* Scales easily. Experts help design heuristics instead of hand labeling. Can be updated easily on all samples
* Algorithms used to combine heuristics during disagreements
* Can use Snorkel.ai library.
* Heuristics are often noisy and don't cover all labels. 

##### Semi-Supervised
* Are many forms of self-supervision
* Requires a small set of initial labels 
* One form is called self-training (outlined below)
* Other methods use clustering or k-nearest
* Perturbation-based method assumes small perturbations should change a label. Small perturbations are added to a sample, and added to the dataset with the same label as the sample
* Self-supervision is most useful when labels are limited
* With limited labels, there's a trade-off between labels used for evaluation and training

##### Self-training
1. Start by training model on existing labels, and then use this model to preidct for all other samples. 
2. Add labels to your data only where this model has output high probability scores
3. Train new model on this expanded dataset. 
4. Repeat until reaching desired performance

#### Transfer Learning
* Is re-using a model developed for one task for another task
* The base task usually has cheap and abundant training data
* In zero-shot learning, the base model is used on the downstream task directly
* Often requires fine-tuning the base model by continuing to train the base model (or part of it) on the downstream task
* Usually the larger the pre-trained base model, the better performance on downstream tasks (e.g. GPT-3)

#### Active Learning
* Chooses which data samples to learn from (often human labeled) to maximize efficiency
* Use metrics / heuristics to dtermine which samples are most helpful to models
	* uncertainty of model on a sample
 	* disagreement between two candidate models
 	* that which reduces loss the most or gives highest gradient updates
 * Can be very helpful in production when data is streaming in real-time and system must choose which samples to label
 	* Because efficiently choosing sample can be effective in dealing with changing environments

### Class Imbalance
* Where substantial difference in number of samples in each class of classifier or when a regression model is highly skewed
* Challenge of insufficient signal from rare classes (becomes few-shot learning)
* Challenge of getting stuck in a non-optimal solution by exploiting a simple heuristic (such as 'always negative')
* Leads to asymmetric costs of error - being wrong on a rare sample distorts model performance a lot
	* i.e. what's the use of an issue detector or cancer identifier that misses positive cases
* Sensitivity to class imbalance increases with complexity of the problem
	* Noncomplex, linearly seperable problems aren't affected by it 
* Binary class imbalance is easier than multiclass class imbalance*
* Deeper neural networks perform better on imbalanced data than shallower

#### Metrics For Class Imbalance
* F1, Precision, Recall are asymmetric methods that depend on which class is considered positive. Should be used for imbalanced problems
* AUC under ROC (Reciever operating curve) is robust to class imbalance since it measures performance at all prediction thresholds
* AUC Under Precision-Recall curve may be even better for class imbalance measurement

#### Resampling
* Oversampling - randomly select samples from rare class to copy and add to dataset
* Undersampling - randomly remove from dominant class
* SMOTE - synthetic minority oversampling technique. Creates new examples of minority class by sampling convex combinations of existing datapoints from it
	* Only proven effective in low-dimensional data
* More sophisitiated sampling techniques require calculating distances between instances which is expensive and hard for high-dimensional data
* Never evaluate on resampled data, only on true distribution
* Two-phase learning - learn on undersampled data, fine-tune model on original data (as in transfer learning)
* Dynamic sampmling - oversample any low-performing classes and undersample high-performing classes during training

#### Algorithm-level Class Imbalance Techniques
* Usually modifies loss functions to deal with class imbalance
* Cost-sensitive learning - uses a cost matrix (same format as confusion matrix) to assign a cost to a given misclassification
* Class-balanced loss- Weights each class in the loss function proportionally by their sample %
* Focal loss - adjust loss so that sample with lower probability of being right has higher loss. Uses a parameter which defines the exponent of (1-P). An exponent of 0 is regular loss

### Data Augmentation
* Used to increase amount of training data. Also hlepful to make models more robust to noise or adversarial attacks
* Standard in CV tasks and now used in NLP tasks
* 3 main types:
1. Label-preserving transformations
2. Perturbations
3. Data Synthesis

#### Label-Preserving Transformations
* In CV, changes image while preserving its label
* Cropping, rotating, flipping, inverting, erasing partly, etc
* In NLP, can randomly replace a word with a similar word which wouldn't change the meaning or sentiment of the sentence

#### Perturbation
* Add some noise to a sample (often image) while preserving its label
* Creates 'adversarial labels'
* The higher resolution an image, the more successful adversarial attacks are
* Helps models to learn weak spots in their decision boundaries and improve performance
* Perturbations can be added randomly or with search
	* DeepFool finds minimum possible noise injection needed to cause a misclassification with high confidence
	* DeepFool is an example of adversarial augmentation

#### Data Synthesis
* In NLP, templates can be used to generate many instances of a common phrase
* Mixup - combines class labels to turn a classification class into a regression task
	* Increases robustness to adversarial examples
* CycleGAN has been shown to improve CV performance in certain tasks

## Feature Engineering




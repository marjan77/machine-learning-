# machine-learning-
Machine learning technique has increased the efficiency in healthcare sector. This has helped to automate a lot of diagnostic procedures. Data related to this sector has beentaken and used in these experiments. The algorithms
compared are Support vector machines, Neural network,
Reinforcement learning, Decision tree, Multi-layered
perceptron, and Naive bayes. Hyperparameter tuning has
also been implemented before passing the parameters to
the algorithms to tune the performance of the algorithms.
Hyperparameter tuning has been done using
GridSearchCV. This is a library in sklearn
model_selection package. Using this will help to select
the best hyperaramets for the model. This algorithm
selects a list of values and is turned as the input for the
parameters. Passing these inputs will return a set of values
which are the best parameters for the estimated model.
It is assumed that Tree-based algorithms will work best
on classification problems. Decision tree is selected along
with other algorithms for experimenting. Hyperparameter tuning is assumed to produce positive results which in
turn will improve the performance the algorithms.
The results for each algorithm are discussed using
confusion matrix and the usage of accuracy, precision,
recall, f1-score and support. The comparison of
algorithms has been done using k-fold cross validation
and plotting ROC curve.
The first dataset contains several health attributes of
people which will predict the chances of heart attack.
Heart attacks are one of the leading causes of heart
disease and the need for early intervention in the event of
a heart attack cannot be underestimated. Thus building a
model to predict chances of heart attack will be of great
help. 
# Results
On the dataset, different models perform at different
levels; nonetheless, we can anticipate which model is
better for a certain type of task. The performance is
determined by various factors, including the nature and
amount of the data, its cleanliness, and the dataset's
linearity.
By working on both datasets, it has been evident that
hyperparameter tuning has not yielded much of a change
in the result. Therefore, the observation is that further
increment of hyperparameters will not capture any great
result and also will lead to usage of more computational
power. Further discussion of result in detail would help us
understand the data better.
The accuracy value of SVM is pretty high in both
datasets. Dataset 1 has an accuracy of 83 and 72 in
Dataset 2. Since accuracy score cannot be fully depended
to check the performance. Since the dataset is imbalanced
other performance scores like recall, precision, and F1
score should also be considered. The resukt for dataset 1
has yielded high scores for all of these performance
scores but in dataset 2 has yielded very poor scores.
The accuracy value for decision tree is 68 for dataset2 and
77 for dataset 1. The hypothesis that tree classifiers will
perform better on classification data has been not
validated using any of the experiment as decision tree
algorithm has performed poorly.
MLP classifier has given accuracy of 85 on dataset 1 and
69 on dataset 2. Other performance scores have also
performed poorly on dataset 2 but has moderate performance on dataset1.
Naïve Bayes is used in NLP for text analysis, yet this
algorithm has produced better results on dataset 1. Recall
(0.84) Precision (0.83) and F1 (0.82). Accuracy on dataset
2 is fairly low which is 74.
KNN which uses a neural network has produced better
results on Dataset1. It has an accuracy of 84. Other
performance scores have yielded good results as well. But
dataset two has performed not that better on KNN.
Reinforcement learning has performed fairly accurate.
The algorithm uses what it has discovered after 2000
steps through exploration in order to raise the average
reward. The final phase was possible because the
algorithm was given time to train the surroundings and
the model is able to balance the pole better than initially
and made sure it stays straight

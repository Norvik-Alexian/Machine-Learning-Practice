# Machine Learning & AI

## Intelligence Definition
There are many definitions of intelligence, there are different types of intelligence defined today.

Intelligence definitions have in common:
* ability to learn
* ability to solve problems
* ability to understand 
* ability to gain and use knowledge
* ability to plan

## AI
AI tries to emulate intelligence, generate model of intelligence.

## AI and Machine Learning
Artificial intelligence refers to machines being able to carry out tasks in a **_smart_** way.

Machine Learning is application of AI based on the concept of providing data to machines and let them learn for themselves.

## AI Subfields
General:
* Learning
* Perception
* Decision Making

## AI definition approaches
* Thinking Humanly
* Thinking Rationally
* Acting Humanly
* Acting Rationally

## Common tasks to solve for AI implementation
* NLP (Natural Language Processing) that allows computer to understand people
* Knowledge representation
* Making Decisions
* Machine Learning to adapt to new data
* Computer Vision
* Robotics

## Machine Learning Problem Types
All Machine Learning algorithms take the data as input, but what they want to achieve is different. \
Machine Learning problems can be classified by following main groups:
* Classification
* Regression
* Clustring
* Association

## Classification
The goal is to classify object (find its class among existent ones).

Input: Object \
Output: Corresponding Class

## Regression
The goal is to predict the value of something. 

Input: Object \
Output: Corresponding numeric value.

## Clustring

Input: Objects \
Output: Groups of Objects

Clusting is the task of grouping together a set of objects in a way that objects in the same cluster are more similar to
each other than to objects in other clusters. \
Similarity is a metric that reflects the strength of relationship between two data objects. \
The goal is to create groups of objects (clusters), discover the inherent groupings in the data.

Clusters can be formed different ways, main clustering solving methods:
1. Connectivity-based
2. Centroid-based
3. Distribution-based
4. Centroid-based

### Clustering Algorithms: Connectivity-based
* Data points that are closer in the data space are more related (similar) than to data points farther away.
* The cluster is formed by connecting data points according to their distance.

### Clustering Algorithms: Centroid-based
* Clusters are represented by a central vector or a centroid.
* This centroid might not necessarily be a member of the dataset.
* This is an iterative clustering algorithms in which the notion of similarity is derived by how close a data point is
to the centroid of the cluster.
* k-means is a centroid based clustering.

### Clustering Algorithms: Distribution-based
* This clustering is very closely related to statistics: distributional modeling.
* Clustering is based on the notion of how probable is it for a data point to belong to a certain distribution, such as
the Gaussian distribution for example.
* Data point in a cluster belong to the same distribution.
* These models have a strong theoritical foundation, however they often suffer from overfitting.

### Clustering Algorithms: Density-based
* Density-based methods search the data space for areas of varied density of data points.
* Cluster are defined as areas of higher density within the data space compared to other regions.
* Data points in the sparse areas are usually considered to be noise and/or border points.
* The drawback with these methods is that they expect some kind of density guide or parameters to detect cluster borders.

### Hard & Soft Clustering
* `Hard clustering` - in hard clustering, each data object or point either belongs to a cluster completely or not.
* `Soft clustering` - in soft clustering, a data point can belong to more than one cluster with some probability or 
likelihood value.

### Choosing Clustering Algorithm
Clutering is a subjective task and there can be more than one correct clustering algorithm. Every algorithm follows a 
different set of rules for defining the 'similarity' among data points. \
The most appropriate clustering algorithm for particular problem often needs to be chosen experimentally, unless there is
a mathematical reason to prefer one clustering algorithm over another. An algorithm might work well on a particular dataset
but fail for a different kind of dataset.

### K-Means Clustering
K-Means falls under category of centroid-based clustering. Taking any two centroid or data points (K=2) in its account
initially. After choosing the centroids, (say C1 and C2) the data points (coordinates) are assigned to any of the clusters
depending upon the distance between them and the centroids. For measuring the distance, you take the following distance
measurement function (similarity measurement function): \
`d = |x2 - x1| + |y2 - y1| + |z2 - z1|` (Manhatan distance) \
where d is distance measurement between two objects, (x1, y1) and (x2, y2) are the X, Y coordinates of any two objects taken
for distance measurement. \
An object which has a shorter distance between a centroid (say C1) than the other centroid (say C2) will fall into the 
cluster of C1. \
There are some metrics for K-means performance measurement:
* Adjusted rand index
* Mutual information based scoring
* Homogeneity, completeness and v-measure.

### Missing Values
Missing values is part of real-life data, not all machine learning algorithms support missing values in the data that 
you are feeding to them. K-Means being one of them, so we need to handle the missing values present in the data. \
There are a couple of ways to handle missing values:
1. Remove rows with missing values
2. Impute missing values.

We can prefer each one depending on one case, for current caae if we remove the rows with missing values it can cause
insufficiency in the data which in turn result in inefficient training of the machine learning model. We will impute
missing values.

There are several ways you can perform the imputation:
* A constant value that has meaning within the domain, such as 0, distinct form all other values.
* A value from another randomly selected record.
* A mean, median or mode value for the column.
* A value estimated by another machine learning model.

Any imputation performed on the train set will have to be performed on test data in the future when predictions are needed
from the final machine learning model.

## SVM
SVM offers very high accuracy compared to other classifiers such as logistic regression, and decision trees. It is used
in a variety of applications such as face detection, intrusion detection, classification of emails, news articles and
web pages, classification of genes, and handwriting recognition.

### SVM principle
The classifier separates data points using a hyperplane with the largest amount of margin. That's why an SVM classifier 
is also known as a discriminative classifier. SVM finds an optimal hyperplane which helps in classifying new data points.

### Simple SVM
There can be several boundaries that correctly divide the data points. The two dashed lines as well as one solid line
classify the data correctly. The most optimal decision boundary is the one which has maximum margin from the nearest 
points of all the classes. The nearest points from decision boundary that maximize the distance between the decision boundary
and the points are called support vectors.

### Kernel SVM
In the case of non-linearly separable data, a stringht line cannot be used as a decision boundary. \
Kernel SVM projects the non-linearly separable data lower dimensions to linearly separable data in higher dimensions in
such a way that data points belonging to different classes are allocated to different dimensions.

### SVM calculations
There is complex mathematics involved behind finding the support vectors, calculating the margin between decision boundary
and the support vectors and maximizing this margin.

## Hyperplane
A hyperplane is a decision plane which separates between a set of objects having different class memberships.

## Margin
A margin is a gap between the two lines on the closest class points. This is calculated as the perpendicular distance from
the line to support vectors or closest points. If the margin is larger in between the classes, then it is considered a good
margin, a smaller margin is a bad margin.

## Support Vectors
Support vectors are the data points, which are closest to the hyperplane. These points will define the separating line 
better by calculating margins. These points are more relevant to the construction of the classifier.

## Data Preprocessing
Data preprocessing involves:
1. Dividing the data into attributes and labels.
2. Dividing the data into training and testing sets.

## Evaluating the Algorithm
Confusion matrix, precision, recall, and F1 measures are the most commonly used metrics for classification tasks.

## MNIST data
The MNIST database is a large database of handwritten digits that is commonly used for training various image processing
systems.

## Normalization (Feature Scaling)
Real world dataset contains features that highly vary in magnitudes, units, and range. Normalization is a technique often
applied as part of data preparation for machine learning. \
The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting 
differences in the ranges of values.

## Feature Scaling Effect
Examples of algorithms where Feature Scaling matters:
* `K-Means` uses the Euclidean measure here feature scaling matters
* `K-Nearest-Neighbours` also require feature scaling.
* `Principle Component Analysis (PCA)` tries to get the feature with maximum variance, here also feature scaling is required.
* `Gradient Descent` calculation speed increase as Theta calculation becomes faster after feature scaling.

## Train & Test Data
`stratify` means that the `train_test_split` method returns training and test subsets that have the same proportions of
class labels as the input dataset.

## Association
Discover rules that describe large portions of your data.

## There are 3 main types of Machine Learning:
* Supervised Learning
* Unsupervised Learning
* Semi-supervised Learning

## Supervised Learning
In Supervised Learning we have possible output for some sample data. \
Thus we have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from 
the input to the output. \
`Y = f(X)` \
The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the
output variables (Y) for that data.

It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of
as a teacher supervising the learning process. \
We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher.\
Learning stops when the algorithm achieves an accpetable level of performance.

In Supervised Learning we need 2 types of data sets:
1. `Training set`: Data with correct/expected output
2. `Test set`: Data on which we should apply our model to test output.

## Supervised Learning Phases
1. `Training`: train model from train data, by pairing the input with expected output.
2. `Validation/Test`: test model on test data to get how well the model works.
3. `Application`: apply model to the real-world data and get the results.

## Unsupervised Learning
Unsupervised learning is where you only have input data (X) and no corresponding output variables. \
The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more
about the data. \
These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is
no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data. \
Algorithm belonging to the family of Unsupervised Learning have no variable to predict tied to the data.

## Semi-supervised Learning
Problem where you have a large amount of input data (X) and only some data is lebeled (Y) are called semi-supervised
learning problems. \
We can use unsupervised learning techniques to discover and learn the structure in the input variables. \
We can also use supervised learning techniques to make best guess predictions for the unlabeled data, feed that data back 
into the supervised learning algorithm as training data and use the model to make predictions on new unseen data.

## NLP (Natural Language Processing)
NLP is one of the most promising fields of artificial intelligence that uses natural languages to enable human interactions
with machines.

## NLP Approaches
* rule-based methods
* learning-based methods

## Text Data & Feature Vector
If data is text, than we should convert it to numeric representation. \
Numeric representation of text is called _**Feature Vector**_. 

To find if texts are similar we can calculate euclidean distance for them.

## Cosine Similarity
Cosine similarity is better to compare words similarity. Cosine similarity is a measure of similarity between two non-zero
vectors of an inner product space that measures the cosine of the angle between them.

## Numpy
Numpy is a library targeted on operations with multi-dimensional arrays (Numeric Python) \
Numpy data structure is called ndarray (ndimensional array)

One-dimensional array - Vector
Two-dimensional array - Matrix

## Linear Regression
Linear regression is a statistical model that examines the linear relationship between two (Simple Linear Regression) or 
more (Multiple Linear Regression) variables - a dependent variable and independent variables.

The relationship between variables is mostly not linear, so there can be some error, the task is to build the line in the
way to minimize the error. When there is one independent variable it's SLR (Simple Linear Regression), often there are
more than one independent variables, it's MLR (Multiple Linear Regression)

## Linear Relationship
Linear relationship basically means that when one independent variables increases, the dependent variable increases too.
Linear relationship can be positive or negative.

Y = a * X + b \
Y - dependent variable (what we want to estimate)
X - independent variable (what we use to make estimation)
a - how much X effect Y (slope)
b - constant (intercept)

## What is pandas?
Pandas is library for working with data objects represented by DataFrames (Panel Data)

## Classification
Classification is identifying the class for new observation based on training data.

## Sigmoid Function (Logistic Function)
Sigmoid function has S-shape and can generate value between 0 and 1

## Logistic Regression
Linear Regression curve can be transformed into Logistic Regression curve. Its very useful to implement LR for binary
classification problems.

Logistic Regression Types:
1. Binary Logistic Regression
2. Multinomial Logistic Regression
3. Ordinal Logistic Regression

## Confusion Matrix Terms
* `True Positives (TP)`: cases in which True was predicted and True was in reality.
* `True Negative (TN)`: cases in which False was predicted and False was in reality.
* `False Positive (FP)`: cases in which True was predicted, but False was in reality. Type 1 Error.
* `False Negative (FN)`: cases in which False was predicted, but True was in reality. Type 2 Error.

## Accuracy & Misclassification Rate
Accuracy = (TP + TN) / Total
Misclassification_rate = (FP + FN) / total

## Bayes Theorem
Method of calculating the validity of beliefs

Original Beliefs + New Evidence = New Stronger Beliefs

Probability of faith validity with new evidence = (Probability of faith validity * Probability of evidence validity in case
of valid faith) / Probability of evidence validity

`P(h | E) = (P(h) * P(E | h)) / P(E)`

## Naive Bayes Classification
Naive Bayes is the most straightforward and fast classification algorithm, which is suitable for a large chunk of data.

Naive Bayes classifier is successfully used in various applications such as spam filtering, text classification, sentiment
analysis, and recommender systems. It uses Bayes theorem of probability for prediction of unknown class.

Naive Bayes is a statistical classification technique based on Bayes Theorem. It is one of the simplest supervised learning
algorithms. \
Naive Bayes classifier is the fast, accurate and reliable algorithm. Naive Bayes classifiers have high accuracy and speed 
on large datasets.

Naive Bayes classifier assumes that the effect of a particular feature in a class is independent of other features. Even 
if these features are interdependent, these features are still considered independently.

Naive Bayes classifier calculates the probability of an event in the following steps:
1. Calculate the prior probability for given class labels
2. Find Likelihood probability with each attribute for each class, then posterior probabilities.
3. Put these value in Bayes Formula and calculate
4. See which class has a higher probability, given the input belongs to the higher probability class.

### Advantages:
* Naive Bayes has very low computation cost.
* It can efficiently work on a large dataset.
* It also performs well in the case of text analytics problems.
* When the assumption of independence holds, a Naive Bayes classifier performs better compared to other models like
logistic regression.

### Disadvantages:
* In practice, it is almost impossible that model will get a set of predictors which are entirely independent.
* Zero Probability / Frequency Problem.

## Classification Workflow
* Learning phase
* Evaluation phase

## Features & Labels
* Feature is input parameter (column).
* Label is output parameter

## Frequency and Likelihood Tables
For simplifying prior and posterior probability calculation you can use the two tables frequency and likelihood tables. \
Both of these tables will help you to calculate the prior and posterior probability. \
The Frequency table contains the occurrence of labels for all features.

There are two likelihood tables. Likelihood Table 1 is showing prior probabilities of labels and Likelihood Table 2 is 
showing the posterior probability.

## Zero Probability Problem
Zero Probability because the occurrence of the particular class is zero, and the model is unable to make prediction. \
The solution for such an issue is the Laplician correction or Laplace Transformation.

## Matplotlib Library
Humans are very visual creatures: we understand things better when we see things visualized.

### matplotlib.pyplot
Library for easy matplotlib usage, it is good to understand it's background work. This is especially handy when you want
to quickly plot something without instantiating any Figures or Axes. \
* `plot()` - generate chart
* `show()` - show chart
* `Figures & Axis` - allow more control

### Figure
* The Figure is the overall window or page that everything is drawn on. It's the top-level component of all the ones that
you will  consider in the following points. You can create multiple independent Figures.
* A Figure can have several other things in it, such as a subtitle, which is a centered title to the figure. You'll also 
find that you can add a legend and color bar, for example, to your Figure.

### Axis
* To the figure you add Axes.
* The Axes are the area on which the data is plotted with functions such as plot() and scatter() and that can have ticks, 
labels, etc. associated with it. This explains why Figures can contain multiple Axes.

### Axis Contents
Each Axes has an x-axis and a y-axis, which contain ticks, which have major and minor tick lines and tick labels.
There's also the axis labels, title, and legend to consider when you want to customize your axes, but also taking into
account the axis scales and gridlines might come in handy.

### Axis Spines
Spines are lines that connect the axis tick marks and that designate the boundries of the data area. In other words, they
are the simple black square that you get to see when you don't plot any data at all but when you have initialized the Axes.

### Subplot 
We use subplot to set up and place your Axes on a regular grid. So that means that in most cases, Axes and subplot are
synonymous, they will designate the same thing. When you do call subplot to add Axes to your figure, do so with the 
`add_subplots()` function.

## What is Decision Tree?
A decision tree is a flowchart-like tree structure where an internal node represents feature (or attribute), the branch
represents a decision rule, and each leaf node represents the outcome.

The topmost node in decision tree is known as the root node.

### Decision Tree Contents
* `Nodes`: Test for the value of a certain attribute.
* `Edges / Branch`: Correspond to the outcome of a test and connect to the next node or leaf.
* `Leaf nodes`: Terminal nodes that predict the outcome.

### Decision Trees Transparency
Decision Trees are easily interpreted and are clear for human, Decision Trees is white box algorithm.

### Where to use Decision Trees
* Categorical Data (Classification)
* Continous Data (Regression)

## Entropy
Shannon invented the concept of entropy as information metrics, which measures the impurity of the input set. \
Entropy (E) correspond to chaos level in system. \
The higher is E the higher is chaos. \
Entropy (E) decrease is information increase.

`E = -Sum(1, N)(p[i] * log2 p[i])`

N is number of possible states of the system. \
p[i] - probability of system to be in state i.

### How to implement Decision Tree
Select the best attribute (question) using Attribute selection measures (ASM) to split the records. Make that attribute a
decision node and breaks the dataset into smaller subsets. Starts tree building by repeating this process recursively for
each child until one of the condition will match:
* All the tuples belong to the same attribute value.
* There are no more remaining attributes.
* There are no more instances.

### Attribute Selection Measures (ASM)
Attribute selection measures is a heuristic for selecting the splitting criterion that partition data into the best 
possible manner. ASM provides a rank to each feature by explaining the given dataset. Best score attribute will be selected
as a splitting attribute. Most popular selection measures are Information Gain, Gain Ratio, and Gini Index.

Information Gain: \
`IG(Q) = E[0] - Sum(1, q)(N[i] / N * E[i])`

Q - is splitting condition (example x <= 12) \
q - number of groups after splitting 
N[i] - number of subset elements

### Optimizing Decision Tree Performance
* **_criterion: optional (default='gini') or choose attribute selection measure:_** This parameter allows us to use the different
attribute selection measure. Supported criteria are "gini" for the Gini index and "entropy" for the information gain.
* _**splitter: string, optional (default="best") or Split Strategy:**_ This parameter allows us to choose the split strategy.
'random' to choose the best random split.
* _**max_depth: int or None, optional (default=None) or Maximum Depth of a Tree:**_ The maximum depth of the tree. If 
None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum
depth causes overfitting, and a lower value causes underfitting.

### Decision Trees Pros & Cons
Advantages:
* Decision trees are easy to interpret and visualize.
* It can easily capture Non-linear patterns.
* Decision trees perform classification without requiring much computation.
* It requires fewer data preprocessing from the user, for example, there is no need to normalize columns.

Disadvantages:
* Sensitive to noisy data, it can overfit noisy data.
* The small variation in data can result in the different decision tree. This can be reduced by bagging and boosting
algorithms.
* Decision trees are biased with imbalance dataset, so it is recommended that balance out the dataset before creating the 
decision tree.

## What is Neural Network?
* Neural Network is one of the methods of Machine Learning.
* Neural Network is build of Artificial Neurons.
* Artificial Neuron is some model based on real neuron.

## Neural Networks Implementation
* NN is implemented for Object Detection, Translation, Text to Speech (Translation Variation)
* Neural Network is implemented when it's hard to find direct solution algorithm. NN will learn to solve the task itself.
* Sometimes there are many correct solutions, every of which is correct for some people and wrong for others.
* Using Neural Network we don't have to know algorithm, we should only know just inputs and outputs.

## Features of Neuron Model
* `Dendrites` - delivers signal to cell, can be connected to axons of other neurons.
* `Axon` - delivers signal from cell, can be connected to dendrites of other neurons, connection is done using Synapses
that can change transferred signal, increase or decrease it.
* `Nucleus` - get an electrical signal from dendrites and after collecting signal nucleus sends some signal to axon.

## Artificial Neuron
Artificial neurons are model that reflects some features considered as important of real neurons.

## Transfer & Activation functions
Transferring function is divide into 2 functions for:
* calculating signal strength usually called as summator function, sometimes as "transfer function"
* calculating if the neuron will send signal based on signal value usually called as activation function.

## Artificial Neural Network
* ANN consists of neurons connected to each other:
* Output signal of one neuron can be transferred to input signal of other neuron.
* ANN consists of 3 kinds of layers: input, hidden, output
  * First layer is called input layer.
  * Last layer is called output layer.
  * Any layer that is between input and output layers is called hidden.

ANN containing more than 1 layer is called _**Deep Neural Network**_. Hidden Layers allows us to make generalizations. 
Generalization can be done for set of neurons from previous layer.

## Types of ANN
There are 2 main types of ANN:
* with direct spreading of signal from input layer to output without cycles.
* recurrent network with possible cycles and possibility to send signal to previous layer or to other neuron in same layer.

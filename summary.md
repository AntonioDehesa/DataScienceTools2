# Week 1: Introduction to Data Science

## Types of learning tasks

* Description: involve the description of patterns and trends in the dataset
* Estimation: to approximate the numeric value of a target or output variable
* Classification: to approximate the categorical value of a target variable 
* Prediction: to approximate the numeric value of a target variable that lie in the future
* Clustering: grouping observations into classes or clusters of similar objects. This is similar to classification, but no target variable is used.
* Anomaly detection: involves predicting if a data point is an outlier compared to other data points in the dataset
* Association: involves finding the relationship between attributes by establishing an if-then rule. It can be used to find out which items in a supermarket are purchased together. 

## Types of learning problems

### Supervised

Uses a training dataset with inputs and outputs to build a model that is later used for predicting the output values of new input data.
Can be further divided into regression and classification.
* Regression: models predict continuous outcomes
* Classification: predict categorical outcomes

Commonly used supervised learning algorithms:
* Decision trees
* random forest
* k-neares neighbor
* support vector machines
* naive bayes
* linear regression
* plynomial regression
* logistic regression
* neural networks

### Unsupervised

involves findind the hidden structure in unlabeled data. 
An example is clustering.

# Week 2 and 3: The Data Science process

Data science workflow: 
1. Data preprocessing
2. Data partitioning
3. model construction
4. model evaluation

## Data preprocessing

It involves preparking the data and making it ready for the modelling. 
It includes:
* data cleaning: remove noisy data, fix inconsistencies, handle missing data and outliers
* transformation: scale the data within a range, such as normalization, transform the data to an appropriate format, etc
* feature extraction: data reduction by deleting redundant or meaningless features
* data exploration: descriptive statistics and visualization

### Data cleaning

it may increase the data quality. garbage in, garbage out.

3 key components of data quality:
* accuracy: inaccurate or noisy data contains errors or values that deviate from the expected
* consistency: inconsistent naming conventions
* completeness: missing values or attrivutes of interest

#### Missing data 

To handle missing data, first you have to make sure there are missing values, which may be present in different formats: blanks, N/A, NA, NAN, 999, -1, etc. 

Once identified, you identify the mechanism in which data is missing. There are 3 mechanisms for patterns of missing data: 
* MCAR: missing completely at random. this is when data is missing completely at random without relation to any other variable
* MAR: missing at random. this means that the data is missing with some relation to one or more variables
* NMAR: not missing at random. Related to one or more variables.

To check if they are MCAR, MAR, or NMAR, you can check the missing values and check their statistics, such as percentages to other values in other variables. 

Once identified, how to handle it:

If less than 5% of cases have missing data and it is either MCAR or MAR, then delete or drop the rows
If more than 5% of cases and MCAR, get the mean, median, mode for imputation.
If more than 5% and MAR, use a regression model to predict missing values
If more than 90% and MCAR, drop the entire variable

#### Inaccurate data

sources of inaccurate data: 
* typographic errors can lead to incorrect values
* misspelled categorical values can create extra values for the variable
* measurement or typographical errors can result to outliers
* duplicate data as it can make that observation to have more influence than it really has

Outliers are unusual and inconsistent data values that deviate remarkably from the expected values. 
They could be valid values, or inaccurate or erroneous data. 
To identify them, numerical and graphical methods are commonly used. 
Numerical methods:
* z-score method: transorm numerical data to z-scores and filter out any z-score above 3
* inter quartile range: IQR = Qe - Q1. Filter out values <(Q1-1.5*IQR) and >(Q3+1.5*IQR)

graphical methods: 
* histogram
* scatter plot
* boxplot

handling outliers: 
you usually just remove them.

### Data exploration

attempt to understand the data using descriptive statistics and visual plots or graphs. 
It can enable us to understand the structure and the distribution of the data, relationships between variables, extreme values, etc. 
Common statistics:
* mean
* mode
* median
* maximum and minimum value
* variance
* standard deviation
* skewness
* kurtosis

the visualization usually includes: 
* histograms
* bar charts
* boxplots
* scatter plots
* scatter matrix

### Data transformation

Data can be transformed by scaling (feature scaling or normalization), discretization (binning), or using numerical values to represent categorical values. 

* scaling: transform values to fall within a samller range, such as -1 to 1, ot 0 to 1, etc. It makes values more easily comparable
There are several ways to perform scaling, such as min-max standardization, z-score standardization, decimal scaling. 

Transformation needs to achieve normality, by reducing skweness. 
Skewness = (3*(mean-median))/standard deviation

* binning: transformation of numerical attributes into categorical attributes. it is achieved by partitioning the numerical attribute values into a finite set of bins. There are several ways to perform binning: 
* equal width binning: creates bins of equal width
* equal frequency binning: creates bins with approximately equal number of data points
* binning by clustering: clusters are used as bins

Avoid converting from categorical to numerical if the numbers have no order, as some algorithms may interpret the numbers as having an order. 
Instead, create dummy variables or one-hot encoding. 

### Feature selection and extraction

Approaches to dimensionality reduction. 
* Feature selection: focuses on selecting a meaningful subset of existing features from the dataset. 
* Feature extraction: focuses on creating a new set of lower dimensional features from the original feature such that the information in the original data is preserved. 

Dimensionality reduction is necessary because: 
* too many features increases run time of algorithms. Sometimes, the algorithm will not even converge. 
* necessary to solve the curse of dimensionality, which is that the greate the number of features, the greater the number of examples needed to train the algorithm. 
* too many features can lead to overfitting, which can be solved with dimensionality reduction. 

In feature selection, only the most informative features are selected. 
It uses techniques such as Principal Component analysis (PCA), Latent Discriminant Analysis (LDA), etc. 
It usually is a linear transformation. 

Sampling can also be used as a data reduction technique. 
It involves randomly selecting a representative subset of the data and using it instead of the entire dataset. 
Useful if the dataset is very large. 

## Data Partitioning

It is the splitting of data into datasets for different purposes.
Instances are randomly assigned to one of the new datasets. 

* two way partition: split dataset into training set (used to develop the model) and test set (used to evaluate the model)
* three way partition: split dataset into training (used to develop models with different hyperparameter settings), test (used to evaluate the performance of the selected model), and validation (used to select best performing model, having the optimal hyperparameter setting)

Percentages recommendations: 
* Training: 60% or 70%
* Validation: 20%
* Test: 20% or 30%

## Model Construction

It involves the application of the appropriate learning algorithm to the training dataset to create the model. 
A model is an abstract representation of the structure or relationships in a given dataset. 
A model can be constructed with different algorithms. 
The model to be constructed is determined by the type of task or problem to be solved.
Supervised:
* classification
* regression

unsupervised: 
* clustering



## Model evaluation

Last step in the data science process. 
This process estimates the performance of a model.
The performance of a model on future or unseen data is called generalization (or test) error. 

This process involves:
* estimates the generalization error
* tuning the hyperparameters of a model to select the proper level of flexibility (model selection)
* comparing algorithms and selecting the best one based on the model performance and algorithm efficiency 

For supervised learning, model performance is estimated using prediction error or accuracy, which involves comparing predicted output values to actual output values. 
Although performance could be estimated for training, validation, or test datasets, it is recommended to use test, as we want to know the performance on new, unseen data. 
If a model performs well on the training dataset, but badly on the test dataset, it is called overfitting.
If it performs badly on both, it is called underfitting. 

The prediction error is high when you have a model with no complexity. 
But there comes a time where model complexity increases so much that it causes overfitting. 

underfitting: model is too rigid to capture the pattern in the data. high bias, low variance
optimal fit: model is not too rigid and not too complex. moderate bias, moderate variance
overfitting: model too complext, and fits all the noise in the data. low bias, high variance

The complexity or flexibility of a model increases with the number of hyperparameters. 
Hyperparameters are parameters that need to be specified a priori.

Model selection involves selecting the best performing model from a given hypothesis space. 
The process of selecting the best performing model, with the optimal hyperparameter value is called hyperparameter tuning. 



### Model evaluation metrics

The ROC curve is a plot of sensitivity vs specificity. 


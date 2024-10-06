# <color> Basics of Machine Learning </color>
## Linear and Logistic Regression

In this project, we wanted to illustrate the application of machine learning on both regression and classification problems.
Therefore, we looked at the basic machine learning models like linear  and logistic regression.


- The project aimed to build the two models from scratch using Python.

- We used data structures, that is, class, which helped in readability and re-usage of the codes.

- We began by defining the Logistic or Linear regression class and then slowly built up on the proceeding functions to complete the two models.
  
## User Instructions
This repository consists of three .py files:

## 1. linear.py

Here we will find the linear logistic Python code built from scratch.

a. Hypothesis
   
Let's begin with the hypothesis

Here, it's easy to observe a linear relationship between our features and targets. So we may settle on the linear function below:

$$y = X\theta$$

where, $X \in \mathbb{R}^{N x D}, \theta \in \mathbb{R}^{D}, y \in \mathbb{R}^N$

b. Criterion/ Loss function
   
Since we have a continuous label, this problem is essentially a regression one and we can use a mean squared error loss.

This is given as $$L(\theta) = \frac{1}{N}∑(y - {\hat y})^2$$
where $y$ is the targets and $\bar y$ is the output of our hypothesis

c. Gradient descent

The idea here is to take little steps in the direction of minimal loss. The gradients when computed guide us in what direction to take these little steps.

A full training loop using a gradient descent algorithm will follow these steps;
- initialize parameters
- Run some number of epochs
  - use parameters to make predictions
  - Compute and store losses
  - Compute gradients of the loss wrt parameters
  - Use gradients to update the parameters
- Do anything else or End!

 ## 2. logistic.py

Here, we will build the logistic regression from scratch using the following mathematical intuition.

a. Logistic/sigmoid function:
$σ(z)= \dfrac{1}{1+ e^{-z}}$

where  $z= x w$.

b. Derivative of Logistic/sigmoid function with respective to $z$:
$σ'(z)= σ(z)(1-σ(z))$

c. Negative log-likelihood or Cross-entropy loss

![math equation](https://quicklatex.com/cache3/e4/ql_95c5f71e61d6398d0bd114805caa3ae4_l3.png)



where:

 $y_{pred}= σ(z)$, $z= xw$.

d. Derivative of Cross-entropy loss with respective to $w$:

$dl(w)= -\dfrac{1}{N}x^T(y_{true} -y_{ped} )$

e. Apply Batch gradient descent to update $w$.

## 3. main.py

- This is the main file where we put our data set values. In our case, we generated the dataset randomly for linear regression, and for classification we used the make classification library.
- You may replace the datasets with your own.
- In this file, we call the two classes LinearRegression and LogisticRegression from the two .py files described above and use the .fit() method to fit our datasets.
- In the def function helps the user to select the model s/he wants to run, whether it is linear or logistic regression. This is just to help in being dynamic instead of running the two models all at once.

## Expected Output
  
- After running your selected model, if it is linear regression, you will get an output of the epochs together with a plot of epoch vs losses which should be decreasing.
- In the case of logistic regression, you will get the losses according to each epoch plus the accuracy percentage.


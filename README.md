# DATA2060_Final

This project implements the multiclass classifiers One vs All and All Pairs algorithms, along with an underlying binary logistic regression function that uses stochastic gradient descent as its optimizer. The logistic regression with stochastic gradient descent can take in different batch sizes and convergence threshold values. For our One vs All and All Pairs algorithms, we can also change the underlying binary logistic regression algorithm to be sklearn's SGDClassifier in order to check predictions and accuracies against our One vs All and All Pairs with our own logistic regression model. The training and test data must have a bias column added to the dataset before inputting into the functions.

To check our model, we developed unit tests for each function/class, and we tested our multiclass algorithms with our own binary logistic regression with stochastic gradient descent on the Iris dataset from sklearn and our Penguins dataset (penguins.csv in our data folder) against sklearn's multiclass classifiers with underlying binary logistic regression SGDClassifier. 

Python version: 3.12.5,
numpy version: 2.0.1,
sklearn version: 1.5.1

Author contacts:
Michael Lu (michael_lu@brown.edu)
Jessica Wan (jessica_wan@brown.edu)
Qiming Fang (qiming_fang@brownn.edu)
Angela Zhu (angela_zhu1@brown.edu)

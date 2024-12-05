import numpy as np

def softmax(x):
    '''
    Apply softmax to an array
    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)

class LogisticRegression:
    '''
    Multiclass Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.
        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_classes, n_features + 1))
        self.alpha = 0.03
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model using stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        converged: bool = False
        num_epochs: int = 0
        while not converged:
            num_epochs += 1

            # Shuffle the data
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            X, Y = X[randomize], Y[randomize]

            # Calculate last epoch loss
            last_loss: float = self.loss(X, Y)

            # Iterate over data in batches to update weights
            for i in range(0, len(X), self.batch_size):
                X_batch, Y_batch = X[i:i + self.batch_size], Y[i:i + self.batch_size]
                gradient = np.zeros_like(self.weights)

                for j in range(X_batch.shape[0]):
                    x, y = X_batch[j], Y_batch[j]
                    pred = softmax(self.weights @ x)
                    gradient += np.outer(
                        pred - (np.arange(self.n_classes) == y),
                        x
                    )
                
                self.weights -= self.alpha * gradient / len(X_batch)

            # Convergence criteria
            if abs(self.loss(X, Y) - last_loss) < self.conv_threshold:
                converged = True

        return num_epochs
    

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of examples.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the average loss of the model on the dataset
        '''
        sum = 0
        probs = [softmax(self.weights @ X[i]) for i in range(len(X))]
        
        # Calculate each sample's loss
        for i, prob in enumerate(probs):
            for j in range(self.n_classes):
                if Y[i] == j:
                    sum -= np.log(prob[j])
        return sum / len(Y)



    def predict(self, X):
        '''
        Compute predictions based on the learned weigths and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        return np.argmax(
            np.array(
              [softmax(self.weights @ X[i].T) for i in range(len(X))]
            ),
            axis=1
        )


    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        return np.mean(self.predict(X) == Y)

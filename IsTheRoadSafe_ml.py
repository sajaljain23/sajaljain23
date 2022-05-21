import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
#Some helper functions for plotting and drawing lines
def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    # print(X[np.argwhere(y==1)])
    rejected = X[np.argwhere(y==0)]
    # print(X[np.argwhere(y==0)])
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')
def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)
data = pd.read_csv(r"C:\Users\hplap\Downloads\data.csv")
# data  = pd.read_csv("bros.csv")

data.columns=[0,1,2]
# print(data[0][0])
X = np.array(data[[0,1]])
# print(X)
y = np.array(data[2])
# print(y)
plot_points(X,y)
# plt.show()


# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):    
    activation = (1+math.exp(-x))**(-1)
    return activation

# Output (prediction) formula
def output_formula(features, weights, bias):
    y = sigmoid(features[0]*weights[0]+features[1]*weights[1] + bias)
    return y

# Error (log-loss) formula
def error_formula(y, output):
    error = -y*math.log(output) - (1-y)*math.log(1-output)
    return error
# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    weights[0] =weights[0] + learnrate*(y- output_formula(x,weights,bias))*x[0]
    weights[1] =weights[1] + learnrate*(y- output_formula(x,weights,bias))*x[1]
    bias = bias + learnrate*(y- output_formula(x,weights,bias))
    return weights, bias


np.random.seed(44)


#try to change the following two values to increase the accuracy
epochs = 100 
learnrate = 0.01
# learnrate=0.1
#leanrate=0.001

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            weights, bias = update_weights(x, y, weights, bias, learnrate)       #update the weights
        
        # Printing out the log-loss error on the training set
        out = output_formula(x, weights, bias)      # calculate the predicted values
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            
            # Converting the output (float) to boolean as it is a binary classification
            # e.g. 0.95 --> True (= 1), 0.31 --> False (= 0)
            predictions = out > 0.5
            
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()


train(X, y, epochs, learnrate, True)

# print(X)
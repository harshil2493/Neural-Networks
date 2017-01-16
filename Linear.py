
# coding: utf-8

# In[5]:

"""
Feed-forward neural networks trained using backpropagation
based on code from http://rolisz.ro/2013/04/18/neural-networks-in-python/
"""


import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

def linear(x) :
    return x

def linear_deriv(x) :
#     return 1
    return np.ones(x.shape)

class NeuralNetwork:
    def __init__(self, layers, activation='tanh') :
        """
        layers: A list containing the number of units in each layer.
                Should contain at least two values
        activation: The activation function to be used. Can be
                "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == 'linear':
            self.activation = linear
            self.activation_deriv = linear_deriv
        self.num_layers = len(layers) - 1
        print 'Layers: ', self.num_layers
        self.weights = [ np.random.randn(layers[i - 1] + 1, layers[i] + 1)/10 for i in range(1, len(layers) - 1) ]
        self.weights.append(np.random.randn(layers[-2] + 1, layers[-1])/10)
#         print 'Weight', self.weights
    def forward(self, x) :
        """
        compute the activation of each layer in the network
        """
        a = [x]
        for i in range(self.num_layers - 1):
#             print 'In Forward, Layer: ', i
            
            a.append(self.activation(np.dot(a[i], self.weights[i])))
        #Adding Linear To Last
        a.append(linear(np.dot(a[self.num_layers - 1], self.weights[self.num_layers - 1])))
            
#         print 'Activation: ', a
        return a
    
    def backward(self, y, a) :
        """
        compute the deltas for example i
        """
        deltas = [(y - a[-1]) * linear_deriv(a[-1])]
        for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
#             if l < len(a) - 2:
            deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
#             else:
#                 deltas.append(deltas[-1].dot(self.weights[l].T)*linear_deriv(a[l]))

        deltas.reverse()
        return deltas
        
    def fit(self, X, y, learning_rate=0.2, epochs=50):
        X = np.asarray(X)
        temp = np.ones( (X.shape[0], X.shape[1]+1))
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.asarray(y)
        

        for k in range(epochs):
            if k%10==0 : print "** ", k, " epochs **"            
            I = np.random.permutation(X.shape[0])
            for i in I :
                a = self.forward(X[i])
                deltas = self.backward(y[i], a)
                # update the weights using the activations and deltas:
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta)
                    
                
    def predict(self, x):
        x = np.asarray(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
#         print 'Self Weight', len(self.weights)
        #Iterate Till Second To Last Layer
        for l in range(0, len(self.weights) - 1):
            a = self.activation(np.dot(a, self.weights[l]))
            
        a = linear(np.dot(a, self.weights[len(self.weights) - 1]))
        return a

def test_digits() :
    
    from sklearn.cross_validation import train_test_split 
    from sklearn.datasets import load_digits
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, mean_squared_error
    from sklearn.preprocessing import LabelBinarizer
    import numpy as np
    digits = load_digits()
    X = digits.data
    y = digits.target
    X /= X.max()
#     network_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    network_values = [2, 4, 8]
#                       , 40, 50, 60, 70, 80, 90, 100]
#     network_values = [64]
#     network_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
#     , 8192, 16384, 32768]
#     , 2048, 4096, 8192]
    initial = 0
#     network_values = [2048]
#     while initial < 205:
#         initial = initial + 5
#         network_values.append(initial)
    print 'Network Paras', network_values
#     network_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print 'X Training Size: ', X_train.shape
    print 'Y Training Size: ', y_train.shape
    print 'X Testing Size: ', X_test.shape
    print 'Y Testing Size: ', y_test.shape
    
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    
    accuracyMean = []
    errorMAD = []
    for index in (network_values) :
        print 'Network Parameter ', index
        nn = NeuralNetwork([64, index, 10],'logistic')
        nn.fit(X_train,labels_train,epochs=100)
        predictions = []
        for i in range(X_test.shape[0]) :
            o = nn.predict(X_test[i])
            predictions.append(np.argmax(o))

        print 'Confusion Matrix'
        print confusion_matrix(y_test,predictions)
        print 'Report'
        print classification_report(y_test,predictions)
        print 'Average'
        print np.mean(precision_score(y_test,predictions, average=None))
        print 'Accuracy'
        print accuracy_score(y_test,predictions)
        accuracyMean.append(np.mean(precision_score(y_test,predictions, average=None)))
        errors = (y_test - predictions)**2
        print 'Calculated Error', 1 - np.mean(precision_score(y_test,predictions, average=None))
        errorMAD.append(1 - np.mean(precision_score(y_test,predictions, average=None)))
#         print 'Y Test', y_test
#         print 'Predictions', predictions
    print 'Network', network_values
    print 'Accuracy', accuracyMean
    print 'Error MAD', errorMAD
#     plot_graph(np.log2(network_values), accuracyMean, 1)
#     plot_graph(np.log2(network_values), errorMAD, 2)   
#     plot_graph((network_values), accuracyMean, 1)
#     plot_graph((network_values), errorMAD, 2)   
def plot_graph(xAxis, yAxis, flag):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(500,500))

    
    
    ax = plt.subplot(111)
    if flag == 1:
        ax.plot(xAxis, yAxis, label = 'Accuracy')
    elif flag == 2:
        ax.plot(xAxis, yAxis, label = 'Error')
    
    for i in range(len(yAxis)):
        plt.plot(xAxis[i], yAxis[i], 'bo')
    ax.legend(bbox_to_anchor=(1.1, 1.05)) 
    
    plt.xlabel('Network Parameters')
    if flag == 1:
        plt.ylabel('Accuracy')
    elif flag == 2:
        plt.ylabel('Error')
#     plt.title('For Iteration: ' + `iteration`)

    plt.show()


    
if __name__=='__main__' :
#     w = np.random.uniform(-1, 1, 2)
#     X = np.random.uniform(-1, 1, [20, 2])
#     y = (np.sign(np.dot(X, w)) + 1) / 2
#     network = NeuralNetwork([2,2,1], 'logistic')
#     network.fit(X,y, epochs=100)
#     for i in range(len(X)) :
#         print i, y[i], network.predict(X[i])

    
    test_digits()


# In[ ]:




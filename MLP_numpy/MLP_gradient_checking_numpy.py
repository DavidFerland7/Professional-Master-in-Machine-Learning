import pickle
import numpy as np
import gzip
import copy

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return train_data, val_data, test_data

# train_data_, val_data_, test_data_ = load_mnist()

class NN_mod_grad_check(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 data=None,
                 init_method="glorot",
                 N =  [1,2,3,4,5,10,20,30,40,50,100,200,300,400,500],
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon
        self.N = N

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data


    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):


            #init of biases
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

            #init of learnable weights
            if self.init_method=="glorot":
                self.weights[f"W{layer_n}"] = np.random.uniform(-np.sqrt(6/(all_dims[layer_n-1]+all_dims[layer_n])),np.sqrt(6/(all_dims[layer_n-1]+all_dims[layer_n])),(all_dims[layer_n-1], all_dims[layer_n]))


    def relu(self, x, grad=False):
        if grad:

            return np.where(x<=0, 0, 1)

        return np.maximum(x, 0)

    def sigmoid(self, x, grad=False):
        sig = 1/(1+np.exp(-x))

        if grad:

            sig = sig*(1-sig)

        return sig

    def tanh(self, x, grad=False):
        _tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        if grad:

            _tanh = 1-np.power(_tanh, 2)

        return _tanh

    def activation(self, x, grad=False):
        if self.activation_str == "relu":

            x_act = self.relu(x, grad)
        elif self.activation_str == "sigmoid":

            x_act = self.sigmoid(x, grad)
        elif self.activation_str == "tanh":

            x_act = self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return x_act

    def softmax(self, x):
        # softmax(x-C) = softmax(x) when C is a constant.

        if len(x.shape)>1:
            a = x - np.max(x, axis=1).reshape(-1,1)
            _softmax = np.exp(a)/np.sum(np.exp(a),axis=1).reshape(-1,1)
        elif len(x.shape)==1:
            a = x - np.max(x)
            _softmax = np.exp(a)/np.sum(np.exp(a))

        return _softmax

    def forward(self, x):
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        cache = {"Z0": x}
        L = len(self.hidden_dims)
        for l in range(1, L + 2):
            cache[f"A{l}"] = np.dot(cache[f"Z{l-1}"], self.weights[f"W{l}"])
            if l <= L:
                cache[f"Z{l}"] = self.activation(cache[f"A{l}"])
            else:
                cache[f"Z{l}"] = self.softmax(cache[f"A{l}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        L = len(self.hidden_dims)

        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        grads[f"dA{L+1}"] = -(labels-output)
        for l in range(L+1, 0, -1):
            grads[f"dW{l}"] = np.dot(cache[f"Z{l-1}"].T, grads[f"dA{l}"])/cache[f"Z{l-1}"].shape[0]
            grads[f"db{l}"] = grads[f"dA{l}"].mean(axis=0).reshape(1,-1)

            if l > 1:
                grads[f"dZ{l-1}"] = np.dot(grads[f"dA{l}"], self.weights[f"W{l}"].T)
                grads[f"dA{l-1}"] = grads[f"dZ{l-1}"]*self.activation(cache[f"A{l-1}"], grad=True)


        self.gradient_check = np.zeros((len(self.N),10,3))
        self.gradient_check_max = np.zeros(len(self.N))

        # finite diff
        backup_weights = copy.deepcopy(self.weights['W2'])

        X_train, y_train = self.train
        for i in range(len(self.N)):
            ##  analytical grad
            self.gradient_check[i,:,0] = grads[f"dW2"][:10,0]
            #print(self.gradient_check)

            # finite diff
            epsilon = 1/self.N[i]

            for j in range(10):
                #print('weights orig: {}'.format(self.weights['W2'][j, 0]))
                self.weights['W2'][j, 0] = backup_weights[j, 0] - epsilon
                #print('weights low: {}'.format(self.weights['W2'][j, 0]))
                #print('backup_weights low: {}'.format(backup_weights[j, 0]))

                train_loss_low, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
                #print('train_loss_low: {}'.format(train_loss_low))

                self.weights['W2'][j, 0] = backup_weights[j, 0] + epsilon
                #print('weights high: {}'.format(self.weights['W2'][j, 0]))
                train_loss_high, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
                #print('train_loss_high: {}'.format(train_loss_high))

                self.gradient_check[i,j,1] = (train_loss_high-train_loss_low)/(2*epsilon)
                #print('self.gradient_check[i,j,1]: {}'.format(self.gradient_check[i,j,1]))
                #print('end round')

            self.gradient_check[i,:,2] = np.abs(self.gradient_check[i,:,0] - self.gradient_check[i,:,1])
            self.gradient_check_max[i] = np.max(self.gradient_check[i,:,2])

        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):

            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - self.lr * grads[f"db{layer}"]

    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        #prediction[np.where(prediction < self.epsilon)] = self.epsilon
        #prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon

        return np.mean(-1*np.sum((np.log(prediction)*labels),axis=1))

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)
                pass

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy


if __name__ == '__main__':
    np.random.seed(1)  ##

    train_data_, val_data_, test_data_ = load_mnist()
    obs_idx = 0
    train_data_1 = (train_data_[0][obs_idx].reshape(1,-1), train_data_[1][obs_idx].reshape(1,-1))
    #valid and test not used, keep only 1 obs anyway to lighten useless calculation
    val_data_1 = (val_data_[0][obs_idx].reshape(1,-1), val_data_[1][obs_idx].reshape(1,-1))
    test_data_1 = test_data_[obs_idx]

    obj_grad_check = NN_mod_grad_check(
        data=(train_data_1, val_data_1, test_data_1),
        hidden_dims=(784, 256),
        epsilon=1e-6,
        lr=7e-2,
        batch_size=64,
        seed=1,
        activation="relu",
        init_method="glorot",
        N = [1,2,3,4,5,10,20,30,40,50,100,200,300,400,500]
    )

    obj_grad_check.train_loop(1)
    print(obj_grad_check.gradient_check)
    print(obj_grad_check.gradient_check_max)
    print(1/np.array([1,2,3,4,5,10,20,30,40,50,100,200,300,400,500]))

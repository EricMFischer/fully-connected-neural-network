import numpy as np
import matplotlib.pyplot as plt
import nndl.layers as layers

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (H, D)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (C, H)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(hidden_size, input_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(output_size, hidden_size)
    self.params['b2'] = np.zeros(output_size)
    '''
    W1: (10, 4)
    b1: (10,)
    W2: (3, 10)
    b2: (3,)
    '''

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape # (5,4): (N,D) (num_inputs,input_size), each row a training ex.
    '''
    W1: (10, 4)
    b1: (10,)
    W2: (3, 10)
    b2: (3,)
    '''

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the output scores of the neural network.  The result
    #   should be (N, C). As stated in the description for this class,
    #   there should not be a ReLU layer after the second FC layer.
    #   The output of the second FC layer is the output scores. Do not
    #   use a for loop in your implementation.
    # ================================================================ #
    
    '''
    Example NN with 3 layers: input (3,1), hidden layers (2,4), output (2,1)
    f = lambda x: x * (x > 0)
    h1 = f(np.dot(W1, x) + b1)
    h2 = f(np.dot(W2, h1) + b2)
    z = np.dot(W3, h2) + b3
    Notes: y = g(z) and f() = ReLU()
    '''

    # H: hidden_size, D: input_size, N: num_inputs, C: num_classes
    # W1 (10,4)(H,D)  DOT  X.T (4,5)(D,N)  =  h1 (10,5)(H,N)

    # Compute the forward pass
    f = lambda x: np.maximum(0, x)
    # h1: for 10 hiddden units, these are 5 input activations
    h1 = f(np.dot(W1, X.T) + b1.reshape(-1,1)) # (10,5): (H,N)
    # z: for 3 classes, these are 5 input scores
    z = np.dot(W2, h1) + b2.reshape(-1,1) # (3,5): (C,N)
    scores = z.T # (5,3): (N,C)
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores # (5,3): (N,C) for 5 inputs, these are 3 class scores

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the loss of the neural network.  This includes the 
    #   softmax loss and the L2 regularization for W1 and W2. Store the 
    #   total loss in the variable loss.  Multiply the regularization
    #   loss by 0.5 (in addition to the factor reg).
    # ================================================================ #
    # scores: (N,C) (num_examples,num_classes)
    # y: [0 1 2 2 1], reg: 0.05

    '''
    loss_dx = layers.softmax_loss(scores, y)
    loss = loss_dx[0] + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    '''

    num_examples = X.shape[0]
    # gets unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (5,3)

    # correct_logprobs: 1D array of each example's correct class probabilities
    correct_logprobs = -1 * np.log(probs[range(num_examples), y])
    
    # compute the loss: average cross-entropy loss and regularization
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss = data_loss + reg_loss
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    grads = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass.  Compute the derivatives of the 
    #   weights and the biases.  Store the results in the grads
    #   dictionary.  e.g., grads['W1'] should store the gradient for 
    #   W1, and be of the same size as W1.
    # ================================================================ #
    
    # probs stores the probabilities of all classes (as rows) for each example.
    # now we can get the gradient on the scores, dscores
    dscores = probs # (5,3)
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    
    # with the gradient on scores, dscores, we can now backpropagate into W and b
    grads['b2'] = np.sum(dscores, axis=0, keepdims=True) # (1,3)
    # grads['W2']: weight gradients: (C,H) for 3 classes, 10 hidden unit gradients
    grads['W2'] = h1.dot(dscores).T + reg * W2 # (H,N).dot(N,C).T (3, 10)
    
    dh1 = dscores.dot(W2) # (5,10) for 5 inputs, the gradient of 10 hidden units
    drelu = (h1.T > 0) * dh1
    
    grads['b1'] = np.sum(drelu, axis=0) # (10,) gradient of b1 for 10 hidden units
    grads['W1'] = X.T.dot(drelu).T + reg * W1 # (10,4) gradient of weights for 10 

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    '''
    X: (5, 4)
    y: (5,)
    X_val: (5, 4)
    y_val: (5,)
    '''
    num_train = X.shape[0] # 5
    iterations_per_epoch = max(num_train / batch_size, 1) # 1

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      # ================================================================ #
      # YOUR CODE HERE:
      #   Create a minibatch by sampling batch_size samples randomly.
      # ================================================================ #
      indices = np.random.choice(num_train, batch_size) # (200,)
      X_batch = X[indices, :] # (200,4)
      y_batch = y[indices] # (200,)

      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      # ================================================================ #
      # YOUR CODE HERE:
      #   Perform a gradient descent step using the minibatch to update
      #   all parameters (i.e., W1, W2, b1, and b2).
      # ================================================================ #
      
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      b2_grad = grads['b2'].reshape(self.params['b2'].shape[0],)
      self.params['b2'] -= learning_rate * b2_grad
      
      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Predict the class given the input data.
    # ================================================================ #

    relu = lambda x: np.maximum(0, x)
    h1 = relu(X.dot(self.params['W1'].T) + self.params['b1'])
    scores = h1.dot(self.params['W2'].T) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return y_pred


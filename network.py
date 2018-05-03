import json
import random
import sys
import numpy as np 

'''
parameters: weights and biases
hyperparameters: learning rate, lmbda, etc.
the network: the "Network" class as it is at that moment
n: generic user input number 
'''


#### Define Quadratic and Cross Entropy Cost Funcitons

''' Purpose of cost function

The cost function is a function of the network's output and correct
label for some given traning example/s. The purpose of this function
is to give the network something quantifiable to improve. Following
this, a cost function should be at a minimum when the network output is
exactly the same as the label, i.e. no error, and increase continuously 
with the error of the output.

With this definition, it's evident that in general the lower the
cost function the better the classification accuracy. This relation, 
however, is not always true and falls apart at higher correct
classification rates and when unseen data is introduced.
'''
class QuadraticCost(object):
	''' Cost is sum of squares of the error
	'''

	@staticmethod
	def fn(a, y):
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(z, a, y):
		return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
	''' Cost is constructed such that the derivative of the cost with respect
	to weighted inputs to the final layer is proportional to the error
	'''
	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(z, a, y):
		return (a-y)

#### Main Network Class
class Network(object):

	def __init__(self, sizes, cost=CrossEntropyCost):			
		# Initializes the Characteristics of the Network

		# defines network shape: a list with the sizes of each layer in order
		self.sizes = sizes
		# number of network layers, including input and output layer
		self.num_layers = len(sizes)

		self.default_weight_initializer()
		# sets the cost function, either Cross Entropy (default) or Quadratic
		self.cost = cost
		# performance of the network, initialized at 0. (successes/trials) ################################################
		self.performance = "0"

	def default_weight_initializer(self):
		""" Initializes weights and biases randomly

		Initializes with a normal distrubtuion, scaled such that the weighted inputs 
		to the next layer has a standard deviation of 2 and therefore avoids the
		learning slowdown cause by the slope of the sigmoid function at abs(z) >> 1

		"""

		# std = 1, x = 0
		self.biases  = [np.random.randn(y, 1) for y in self.sizes[1:]]
		# std = 1/sqrt(len(next_layer)), x = 0
		self.weights = [np.random.randn(y, x)/np.sqrt(x) 
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def large_weight_initializer(self):
		""" Initializes weights and biases unidealy

		Initializes each weight and bias with a normal distribution of std = 1, x = 0
		Result is very large weighted inputs and therefore learning slowdown

		"""
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y, x) 
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def feedforward(self, a): 			
		# Return the output of the network for input 'a'

		for b, w in zip( self.biases,self.weights ):
			a = sigmoid( np.dot(w, a) + b )
			
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, 
			lmbda = 0.0,
			test_data = None,
			evaluation_data = None,
			monitor_evaluation_cost = False,
			monitor_evaluation_accuracy = False,
			monitor_training_cost = False,
			monitor_training_accuracy = False,
			keep_best = False,
			show_progress = False,
			early_stopping_n = 0,
			eta_schedule_change_n = 0,
			eta_decrease_factor = 2,
			eta_decrease_steps = 0): 

		""" Trains network using Stochiastic Gradient Descent
		
		Incrementally improves the network by adjusting the weights and biases to better 
		fit 'minibatches' of multiple training examples using gradient descent. 
		These minibatches are small data sets randomly selected from the full training set. 
		Once all data from the full training set has been selected and trained on,
		the network can be re-trained over the same training data but with differently
		selected minibatches. 

		There is the option to display the network's performance in various metrics after 
		each Epoch to keep track of progress. It's also possible to use this tracking to 
		apply early stopping and learning rate scheduling.

		Args:
			training_data: list of tuples "(x,y)" where x is the input 
				and y is the correct output
			epochs: maximum number of epochs i.e. number of interations over all training data
			mini_batch_size: number of training examples to batch
			eta: learning rate
			lmbda: normalization factor ########################## DESCRIPTION ABOUT HOW TO ADJUST FOR DIFFERENT DATASIZES OR MINIBATCHES??

		Returns:
			Lists containing performance of every tracked metric after each epoch of training
		"""


		# converts data from unusable tuple to usable list
		#training_data = list(training_data)
		n = len(training_data)


		# main loop which trains the network for epochs
		for j in range(epochs):
			# randomly creating all mini_batches for this epoch
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			# updates weights and biases incrementally for every mini_batch	
			for mini_batch in mini_batches:
				self.update_mini_batch(
					mini_batch, eta, lmbda, n, mini_batch_size)


	def update_mini_batch(self, mini_batch, eta, lmbda, n, mini_batch_size): 
		''' Update network parameters based on a single minibatch
		
		Uses gradient descent in the self.backprop method to find the gradient 
		of the cost function for a single minibatch, adds a regulatization factor,
		then updates the weights and biases.

		Args:
			mini_batch: list of tuples "(x,y)" where x is the input 
				and y is the correct output
			eta: learning rate
			lmbda: regularization factor

		Returns:
			None: updates weights and biases 
		'''

		# initializes array containing the sum of gradients of each individual training example
		nabla_b = [ np.zeros( b.shape ) for b in self.biases ]
		nabla_w = [ np.zeros( w.shape ) for w in self.weights ]
		
		# creates gradient array from sum of individual gradients
		for x, y in mini_batch:
		# find gradient of a single training example using backpropogation
			delta_nabla_b, delta_nabla_w = self.backprop( x, y )

			nabla_b = [ nb + dnb for nb, dnb in zip( nabla_b, delta_nabla_b ) ]
			nabla_w = [ nw + dnw for nw, dnw in zip( nabla_w, delta_nabla_w ) ]
		
		# updates parameters using the gradients based on the function v' = v - (eta/n)*nabla_v
		self.biases  = [ b - ( eta/mini_batch_size ) * nb 
						for b, nb in zip( self.biases,  nabla_b)  ]
		self.weights = [ (1-eta*(lmbda/n))*w - ( eta/mini_batch_size ) * nw 
						for w, nw in zip( self.weights, nabla_w)  ]
		
	def backprop(self, x, y): 
		''' Finds the gradient of the cost function

			Calculates the gradient of the cost function with respect to the 
			current weights and biases for a single training example.

			**LONG DESCRIPTION

			Args:
				x: single training input
				y: desired output for input 'x'

			Returns:
				nabla_b: bias gradients
				nabla_w: weight gradients
		'''

		# initializes containers
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		weighted_inputs = []

		# forward propogation with [z] tracking
		# Tracking the weighted inputs and their gradients are critical
		# for simpler and faster calculations.
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			activation = sigmoid(z)
			# saves- 'z' and 'a'
			weighted_inputs.append(z)
			activations.append(activation)

		# derivative of cost function with respect to the weighted input of output layer
		# delta contains the gradient of the weighted inputs of the 'current' layer
		delta = (self.cost).delta(weighted_inputs[-1], activations[-1], y)
		# saves gradients of the parameters of the last layer - not in loop because delta is defined differently to the other layers
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# Loops through the hidden layers in reverse order, starting from the last hidden layer
		for l in range(2, self.num_layers):
			z = weighted_inputs[-l]
			# delta(l) = w(l+1)jk * delta(l+1)  *  sig'(z(l)) 
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
			# save gradients for current layer
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def feedforward_softmax_output(self, a, hardness): 			
		''' Return the softmax output of the network for input 'a'
		Softmax outputs are a probability distribution, where every output
		represents the probability of that neuron being the correct classification. 
		It does this by normalizing the exponential of each weighted input 
		so that they all add to 1. 

		hardness: parameter used to change the exponent's base
				  (e^z)^hardness
		'''
		n = 0
		for b, w in zip( self.biases,self.weights ):
			z = np.dot(w, a) + b
			n += 1
			if n == self.num_layers - 1:
				break
			a = sigmoid(z)
		z_exp = np.exp(z*hardness)
		normal = np.sum(z_exp)
		confidence = z_exp/normal
		return confidence

	def feedforward_softplus_output(self, a):
		'''Return the output of the network for input 'a'
		caution: no backpropogation support
		'''
		for n in range(self.num_layers-2):
			b = self.biases[n]
			w = self.weights[n]
			a = sigmoid(np.dot(w, a) + b)
		b = self.biases[self.num_layers-2]
		w = self.weights[self.num_layers-2]
		#y = relu(np.dot(w, a) + b)
		z = np.dot(w, a) + b
		z = float(z[0][0])
		return softplus(z)

	def keep_best(self):
		# Saves best parameters for file saving once training is complete 
			self.biases = self.best_biases
			self.weights = self.best_weights

	def save_performance(self, test_data, n_test):
		# Saves performance on test_data  for file saving once training is complete
		performance = self.accuracy(test_data)
		self.performance = performance/n_test
		print("Test accuracy for best evaluation accuracy: {}/ {}".format(performance, n_test))

# data label helper
def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e

# element-wise sigmoid activation function
def sigmoid(z):						
	return 1.0/(1.0+np.exp(-z))

# element-wise relu activation function
def softplus(z):
	return np.log1p(np.exp(z))

# element-wise derivative of sigmoid function
def sigmoid_prime(z):				
	return sigmoid(z)*(1-sigmoid(z))

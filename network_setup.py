import numpy as np
from random import seed

def init_network(num_inputs: int,
				 num_hidden_layers: list[int],
				 num_nodes_hidden: int,
				 num_nodes_output: int) -> dict:
	"""
	Create a dict of a network with set number of layers and nodes, 
	initialising with random values
	"""	

	network = {}  # Empty dict to store network
	num_nodes_previous = num_inputs  # Needed initialising for loop

	# Each layer: initialise layer_names, nodes,  weights and biases 
	for layer in range(num_hidden_layers + 1):
		# Output layer name and nodes
		if layer == num_hidden_layers:
			layer_name = 'output'
			num_nodes = num_nodes_output
		# Other layer names and nodes
		else:
			layer_name = 'layer_{}'.format(layer + 1)
			num_nodes = num_nodes_hidden[layer]

		# Initalise wweights and biases
		network[layer_name] = {}
		for node in range(num_nodes):
			node_name = 'node_{}'.format(node+1)
			network[layer_name][node_name] = {
            	'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            	'bias': np.around(np.random.uniform(size=1), decimals=2),
       		 }
	
		num_nodes_previous = num_nodes
	
	return network


def apply_weighted_sum(inputs, weights, bias):
	"""
	Weighted sum for a node
	"""

	return np.sum(inputs * weights) + bias


def apply_node_activation(weighted_sum):
	"""
	Apply activation function to weighted sum of node

	Sigmoid function
	"""	

	return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


def apply_forward_prop(network, inputs):
	"""
	Forward propogation of a network:

	1. input layer -> weighted sum at nodes -> node outputs
	2. node_output_prev -> input to next layer
	repeat 1-2 for next layer
	terminate at output layer.
	"""

	layer_inputs = list(inputs)  # Input layer is input for first hidden layer
	
	for layer in network:
		
		layer_data = network[layer]
		
		layer_outputs = []
		for layer_node in layer_data:
			node_data = layer_data[layer_node]
			
			# Compute weighted sum and output of each node
			node_weighted_sum = apply_weighted_sum(layer_inputs, 
												   node_data['weights'],
												   node_data['bias'])
			node_output = apply_node_activation(node_weighted_sum)
			layer_outputs.append(np.around(node_output[0], decimals=4))
			
		if layer != 'output':
			print(f'Outputs of nodes in hidden layer {layer.split('_')[1]}: {layer_outputs}')

		layer_inputs = layer_outputs  # Setting output of layer to be input of next layer
	
	network_predictions = layer_outputs
	return network_predictions
	
# Initialise network
network = init_network(num_inputs = 5,
				   	   num_hidden_layers = 3,
				   	   num_nodes_hidden = [3,2,3],
				   	   num_nodes_output = 1)

# Create inputs
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

# Forward propogate
predictions = apply_forward_prop(network, inputs)

print(predictions)

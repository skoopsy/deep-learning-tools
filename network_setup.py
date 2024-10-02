import numpy as np

def init_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
	

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


num_inputs = 2 # number of inputs
num_hidden_layers = 2 # number of hidden layers
num_nodes_hidden  = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer	

print(init_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output))


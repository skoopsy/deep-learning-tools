import numpy as np

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


# Test
network = init_network(num_inputs = 5,
				   	   num_hidden_layers = 3,
				   	   num_nodes_hidden = [3,2,3],
				   	   num_nodes_output = 1)

print(network)


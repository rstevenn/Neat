import pickle
import random as rd

class Node:
    """
    A Neural network's node
    """
    def __init__(self, index, node_type, activation_fn=lambda x: max(0, x)):
        """
        index: the index o the node in the neural network (int) (> 0)
        node_type: the type of the node (str) (bias, input, output, hidden)
        activation_fn: the activation fonction of the node (fonction)
        """

        self.index = index
        self.node_type = node_type
        self.activation_fn = activation_fn
        self.root = []
        self.input_links = []
        self.output_links = []
        self.value = 0

    def activate(self):
        """
        Activate the node based on the value of the input nodes
        """
        self.value = 0
        
        for link in self.input_links:
            self.value += link.input_node.value * link.weigth

        self.value = self.activation_fn(self.value)

    def to_dict(self):
        """
        Convert the class in a basic dictionary 
        """
        return dict(inputs = list(link.input_node for link in self.input_links),
                    outputs = list(link.output_node for link in self.output_links),
                    index = self.index,
                    type = self.node_type,
                    activation = self.activation_fn)

class Link:
    """
    A link between 2 nodes
    """
    def __init__(self, innovation, input_node, output_node, weigth=(2 * rd.random() - 1)):
        """

        """
        
        self.innovation = innovation
        self.input_node = input_node
        self.output_node = output_node

        self.weigth = weigth

class NN:
    pass

class Speice:
    pass

class Neat:
    pass
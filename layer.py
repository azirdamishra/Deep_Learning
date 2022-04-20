class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        #return the output of calcs
        pass

    def backward(self, output_gradient, learning_rate):
        #update params and return input gradients 
        pass
import numpy as np
def forwardPass(inputs, weight, bias):
    w_sum= np.dot(inputs,weight)+bias
    #linear activationf(x)=x
    act = w_sum
    return act
#pre-trained weights&biasses after training
W = np.array([[2.99999928]])
b =np.array ([1.99999976])
#instalize input data
inputs =np.array([[7],[8],[9],[10]])
#output layer
o_out = forwardPass(inputs, W, b)
print('Output layer(linear)')
print('=====================')
print(o_out,"\n")
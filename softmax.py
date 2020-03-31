import torch

# We will write a function that takes as input a list of numbers, and returns
# the list of probability values given by the softmax function.
def softmax(L):
    return torch.exp(L)/torch.sum(torch.exp(L), dim=1).view(-1, 1) 
#torch.sum(torch.exp(), dim=1) this takes the exponents of all the values and sum up by column
#.view(-1, 1) this changes the rows and columns (transpose) to enable the tensor division (matrix inverse multiplication)

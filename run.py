from ngramModule import NGram
import numpy as np
import json
from tqdm import tqdm

# Reading the data
with open('data.json','r') as f:
    data = json.load(f)

# Counting the tokens and returning the sorted set
def countTokens() -> set:
    tokens : list = []
    for entry in data["Data"]:
        tokens.extend(entry["Text"].split(" "))
    return sorted(set(tokens))
tokens = countTokens()

# Shortcut for calculating the cross entropy derivative
def crossEntropyShort(x,predIndex):
    error = x - 1
    error[predIndex] +=1
    return error

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x, axis=0)

# Xavier's intialization
def intializeMatrices(n,m):
    return np.random.uniform(-1,1,(n,m)) *  (6/(m+n))**0.5

# Similarity head using jacard similarity
class SimilarityHead:
    def __init__(self,n,tokens) -> None:
        self.n = n
        self.gram = NGram(n)
        self.tokens = tokens

    def compute(self, word):
        result = np.zeros((len(self.tokens),1))
        for i,token in enumerate(self.tokens):
            # Finds similarity with each word in the token corprus
            result[i] = self.gram.findSimilarity(word,token)

        return result

# Multihead similarity
class MultiHead:
    def __init__(self, headCount, tokens) -> None:
        self.headCount = headCount
        self.heads = []
        self.tokens = tokens
        # Increments the gram size as we add more heads
        # You can use any string smilarity algorithm
        # Additionally you can use the same algorthim but use varied parameters
        for i in range(1, headCount+1):
            self.heads.append(SimilarityHead(i,tokens))

    def compute(self,word):
        # Uses the first head to compute base matrix
        result = self.heads[0].compute(word)
        # Loops through all heads
        for head in self.heads[1:]:
            # Concats each result to the main matrix
            result = np.concatenate((result,head.compute(word)),axis=1)
        return result
    
# The corrector network itself
class Network:
    def __init__(self, numberOfHeads, tokens, hiddenSize, learningRate) -> None:
        # Creates multihead object
        self.heads = MultiHead(numberOfHeads,tokens)
        self.tokens = tokens
        self.learningRate = learningRate
        self.headCount = numberOfHeads
        # Intializes matrix U using xavier intialization
        # Matrix U is of size HxS 
        self.U = intializeMatrices(numberOfHeads,hiddenSize)
        # Initializes vector V using xavier intialization
        # Vector V is of size Sx1
        self.V = intializeMatrices(hiddenSize,1)

    def forward(self, input):
        # Computes the concated head matrix
        # The resulting matrix is of size NxH
        self.headOutput = self.heads.compute(input)
        # Computes the dot product of the head matrix and matrix U
        # The resulting matrix is of size NxS
        self.outXU = np.dot(self.headOutput,self.U)
        # Computes the dot product of matrix outXU and vector V
        # The resulting vector is of size Nx1 
        self.outXV = np.dot(self.outXU,self.V)
        # Runs a softmax function over the outputs
        self.output = softmax(self.outXV)
        return self.output
    
    def backward(self,error):
        # Computes gradient for vector V
        # ∂L/V = ∂L/O , ∂O/outXV . ∂outXV/V
        # Gradient is the result of the dot product of the partial derivative of outXV to V, which is outXU transposed, with the error
        # We use dot product to get an output vector of size Sx1
        gradV = np.dot(self.outXU.T,error)
        # Computes gradient for matrix U
        # ∂L/U = ∂L/O . ∂O/outXV . ∂outXV/outXU . ∂outXU/U
        # Gradient is the result of the dot product of the partial derivative of outXU to U, which is the head matrix transposed, with the
        # dot product of error with the partial derivative of outXV to outXU, which is the V vector transposed
        # We use dot product to get an output matrix of size HxS
        gradU = np.dot(self.headOutput.T,np.dot(error,self.V.T))

        # Then we update the weights using the computed gradients
        self.U -= self.learningRate*gradU
        self.V -= self.learningRate*gradV

# Creates a newtwork object
corrector = Network(3,tokens,100,0.2)
# Gets the index of the word 'helped'
i = list(tokens).index('helped')
# Makes a prediction
res = corrector.forward('helped')
# Calculates the error
error = crossEntropyShort(res,i)
# Runs a backward pass
corrector.backward(error)

# Loops for 20 times
# For some reason when we train it on one input it gets trained for all the others
# My theory is the similarity algorithms are doing most of the work we just need to enhance their work
for _ in tqdm(range(20)):
    # Makes a prediction
    res = corrector.forward('helped')
    # Calculates the error
    error = crossEntropyShort(res,i)
    # Runs a backward pass
    corrector.backward(error)

# Takes user input for sentence
text = input('Enter sentence: ')
for word in text.split(' '):
    if word not in tokens:
        # Makes a prediction if the word is novel to the model
        res = corrector.forward(word)
        print(tokens[np.argmax(res)],end=' ')
    else:
        print(word,end=' ')
print()


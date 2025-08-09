# Numpy Implementation of RNN

This project is a from-scratch implementation of a character-level Recurrent Neural Network (RNN) using only the numpy library. It is trained on the source code of the Python requests library and learns to generate new Python code one character at a time. 

## Project Overview
The goal of this project is to build a generative language model from fundamental principles, without relying on deep learning frameworks like PyTorch or TensorFlow. By training on a specific domain—Python source code—the model learns the syntax, structure, and common patterns of the language, enabling it to generate coherent (though not always perfect) code snippets.

### The Goals of This Project.
1. Understanding the RNN architecture.
2. Implementation of BTT in Numpy

### Limitations
1. it doesn't quite work. This is due to the limitations of the base level rnns on large bodies of text. It faces the typical gradient vanishing / exploding problems, that can be solved using another architecture like LSTMs or Gated Recurrent Units (GRU), which were developed to mitigate these issues and would be a logical next step for improving performance on this project.

### Methodology
1. Model Architecture

The model is a classic vanilla RNN composed of three main parts:

**Embedding Layer**: A learnable lookup table (E) that maps each character in the vocabulary to a dense vector representation. 

**Recurrent Hidden Layer**: The core of the RNN. The hidden state $(h_t)$ at each time step is computed based on the current input character's embedding $(x_t)$ and the previous hidden state $(h_(t−1))$. The tanh activation function is used. The update rule is:

$h_t=tanh(W_xhx_t+W_hhh_t−1+b_h)$

Output Layer: A fully connected layer that maps the hidden state to a vector of logits the size of the vocabulary. A softmax function is then applied to these logits to produce a probability distribution over the next possible character.

$y_t=softmax(W_hyh_t+b_y)$

2. Data Preparation & Training
Dataset: The training data was created by cloning the official GitHub repository for the requests library and concatenating all .py files into a single text block. 

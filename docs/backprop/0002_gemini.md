This guide provides a detailed breakdown of the four core equations of
backpropagation and their associated symbols.

Backpropagation is supervised learning's primary method for computing the
gradient of a loss (or cost) function with respect to the weights and biases of
a neural network. It applies the chain rule of calculus systematically, moving
backward from the output layer to the input layer.

The Fundamental Definition of "Error" (\delta)

Before analyzing the equations, it is helpful to understand the central
variable: \delta_j^l (the error of neuron j in layer l).

Mathematically, this error is defined as:
\delta_j^l \equiv \frac{\partial C}{\partial z_j^l}

This derivative measures how sensitive the overall cost C is to changes in the
pre-activation input z_j^l of that neuron. If \frac{\partial C}{\partial z_j^l}
is large, then small changes to this neuron's input will significantly change
the final cost. Backpropagation is essentially a set of rules for calculating
this \delta for every neuron, which then makes finding the actual gradients
straightforward.

Detailed Explanation of the Four Equations

Equation 1: Output Layer Error

\delta^L = \nabla_a C \odot \sigma'(z^L)

  - Purpose: Computes the error vector (\delta^L) for the final layer (L) of the
    network. This is the starting point of the backward pass.
  - How it works:
      - \nabla_a C (Gradient of Cost w.r.t. Output Activations): This measures
        how fast the cost changes as a function of the output activations a^L.
        For example, if we use Quadratic Cost (Mean Squared Error) where
        C = \frac{1}{2} \sum (a^L - y)^2, then \nabla_a C = (a^L - y).
      - \sigma'(z^L) (Activation Derivative): This measures how fast the
        activation function \sigma is changing at the pre-activation value z^L.
      - \odot (Hadamard Product): This denotes element-wise multiplication. We
        multiply the output sensitivity vector by the activation derivative
        vector element-by-element. If a neuron's activation is in a "flat"
        region of its curve (where \sigma'(z^L) \approx 0), the error becomes
        very small, a phenomenon known as saturated neurons.

Equation 2: Hidden Layer Error (The Backpropagation Step)

\delta^l = \bigl( (w^{l+1})^T \delta^{l+1} \bigr) \odot \sigma'(z^l)

  - Purpose: Computes the error (\delta^l) at any hidden layer l using the known
    error (\delta^{l+1}) from the next layer.
  - How it works:
      - (w^{l+1})^T \delta^{l+1}: This term uses the transpose of the weight
        matrix for layer l+1. By multiplying (w^{l+1})^T by the error vector
        \delta^{l+1}, we are effectively moving the error backward through the
        weights. This calculates how much each neuron in layer l contributed to
        the errors in layer l+1.
      - \odot \sigma'(z^l): Just like in Equation 1, we multiply element-wise by
        the derivative of the activation function at layer l. This scales the
        backpropagated error based on how responsive the activation function is
        at that layer's current state.

Equation 3: Gradient with Respect to Biases

\frac{\partial C}{\partial b_j^l} = \delta_j^l

  - Purpose: Computes the partial derivative of the cost function with respect
    to any bias b_j^l in the network.
  - How it works:
      - The bias term b_j^l enters the equation for the pre-activation linearly:
        z_j^l = \sum (w_{jk}^l a_k^{l-1}) + b_j^l.
      - Because \frac{\partial z_j^l}{\partial b_j^l} = 1, applying the chain
        rule yields:
        \frac{\partial C}{\partial b_j^l} = \frac{\partial C}{\partial z_j^l} \frac{\partial z_j^l}{\partial b_j^l} = \delta_j^l \cdot 1 = \delta_j^l
      - This means the gradient for any bias is exactly equal to the error
        \delta_j^l calculated for that neuron.

Equation 4: Gradient with Respect to Weights

\frac{\partial C}{\partial w_{jk}^l} = a_k^{l-1} \delta_j^l

  - Purpose: Computes the partial derivative of the cost function with respect
    to any individual weight w_{jk}^l.
  - How it works:
      - The weight w_{jk}^l links the activation of neuron k in the previous
        layer (a_k^{l-1}) to the input of neuron j in the current layer (z_j^l).
      - Using the chain rule:
        \frac{\partial C}{\partial w_{jk}^l} = \frac{\partial C}{\partial z_j^l} \frac{\partial z_j^l}{\partial w_{jk}^l}
      - Since z_j^l = \dots + w_{jk}^l a_k^{l-1} + \dots, the derivative
        \frac{\partial z_j^l}{\partial w_{jk}^l} is simply the incoming
        activation a_k^{l-1}.
      - Substituting this back in gives a_k^{l-1} \delta_j^l.
      - Practical consequence: If the incoming activation a_k^{l-1} is very
        close to zero, the weight gradient will also be close to zero, meaning
        the weight will learn very slowly.

Grouped Symbol Glossary

To make the notation easier to parse, here are the symbols grouped by their role
in the network.

Indices and Dimensions

  - L: The index representing the output layer.
  - l: A general index representing any given layer in the network (where l = 1
    is the first hidden layer, and l = L is the output layer).
  - j: The index of a specific neuron in the destination layer (l).
  - k: The index of a specific neuron in the source/previous layer (l-1).

Core Variables (Vectors and Matrices)

  - z^l (Weighted Input): The vector of raw, pre-activation values for layer l.
    Mathematically: z^l = w^l a^{l-1} + b^l.
  - a^l (Activation): The vector of outputs from layer l after the activation
    function is applied: a^l = \sigma(z^l).
  - w^l: The weight matrix for layer l. The element w_{jk}^l represents the
    weight connecting neuron k in layer l-1 to neuron j in layer l.
  - b^l: The bias vector for layer l.
  - \delta^l (Error Vector): The sensitivity of the cost function to the
    pre-activation values of layer l.

Functions and Operators

  - \sigma: The activation function (such as Sigmoid, Tanh, or ReLU).
  - \sigma': The first derivative of the activation function.
  - \nabla_a C: The gradient of the cost function with respect to the output
    activations. It is a vector of partial derivatives:
    \left[ \frac{\partial C}{\partial a_1^L}, \frac{\partial C}{\partial a_2^L}, \dots \right]^T.
  - \odot (Hadamard Product): An operator indicating element-wise multiplication
    of two matrices or vectors of the same dimensions. For example:
    \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \odot \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 y_1 \\ x_2 y_2 \end{bmatrix}

Step-by-Step Execution of Backpropagation

Using these terms, the standard workflow for training a neural network on a
single training instance follows these steps:

1.  Forward Pass: Set the input activation a^1. For each subsequent layer
    l = 2, 3, \dots, L, compute z^l = w^l a^{l-1} + b^l and a^l = \sigma(z^l).
2.  Output Error: Compute the error vector at the final layer, \delta^L, using
    Equation 1.
3.  Backpropagate Error: For each layer l = L-1, L-2, \dots, 2, compute the
    error vector \delta^l using Equation 2.
4.  Compute Gradients: Use Equation 3 and Equation 4 to calculate how the cost
    function changes with respect to every bias and weight in the network. Use
    these gradients to update the weights and biases via your chosen
    optimization algorithm (e.g., Stochastic Gradient Descent).

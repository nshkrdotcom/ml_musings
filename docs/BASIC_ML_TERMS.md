In machine learning literature, papers, and code, there is a widely accepted set of mathematical notations and conventions. Understanding these standard symbols makes it much easier to read ML formulas and understand how data flows through a model.

Here are the most common basic conventions, organized by their role in the machine learning pipeline.

---

### 1. Inputs, Outputs, and Predictions
These variables describe the data itself.

*   **$x$ (or $\mathbf{x}$): Input**
    *   A single input sample (usually a vector).
*   **$X$ (or $\mathbf{X}$): Batch of Inputs**
    *   Capitalization usually indicates a matrix or a higher-dimensional tensor. $X$ represents a batch of multiple input samples.
*   **$y$ (or $\mathbf{y}$): Target (Ground Truth)**
    *   The "correct answer" or label we want the model to learn to predict (e.g., the actual price of the house, or the correct class of an image).
*   **$\hat{y}$ (pronounced "y-hat"): Prediction**
    *   The output actually produced by the model. The hat symbol $(\,\hat{}\,\)$ in statistics almost always denotes an *estimate* or *prediction*. The goal of training is to make $\hat{y}$ as close to $y$ as possible.

---

### 2. Learnable Parameters
These are the internal values that the model adjusts during training to improve its predictions.

*   **$W$ (or $\mathbf{W}$): Weights**
    *   A matrix of numbers that determines how much influence each input feature has on the output. It scales the inputs.
*   **$b$ (or $\mathbf{b}$): Bias**
    *   A vector added to the weighted inputs. It allows the model to shift its predictions up or down, independent of the input values.
*   **$\theta$ (pronounced "theta"): All Parameters**
    *   Often used as a shorthand to represent all of the model's learnable parameters combined (both weights and biases: $\theta = \{W, b\}$).

---

### 3. Intermediate Layers (Hidden States)
As data passes through a multi-layer neural network, it goes through intermediate steps.

*   **$z$ (or $\mathbf{z}$): Pre-activation Value**
    *   The raw linear combination before an activation function is applied: $z = Wx + b$.
*   **$a$ or $h$ (or $\mathbf{a}, \mathbf{h}$): Activation / Hidden State**
    *   The output of a hidden layer after applying the non-linear activation function: $h = \sigma(z)$.
    *   "$h$" stands for *hidden*, and "$a$" stands for *activation*.

---

### 4. Functions and Optimization
These symbols represent the mathematical operations that evaluate and improve the model.

*   **$\sigma$ (sigma) or $f$: Activation Function**
    *   A non-linear function applied to the hidden layers (e.g., ReLU, GeLU, Sigmoid).
*   **$\mathcal{L}$ or $L$: Loss Function**
    *   A function that measures the error for a *single* sample (how far $\hat{y}$ is from $y$).
*   **$J$ or $E$: Cost / Objective Function**
    *   The average loss calculated over the *entire dataset* or batch.
*   **$\eta$ (eta) or $\alpha$ (alpha): Learning Rate**
    *   A small scalar value (e.g., 0.001) that controls how large of a step the model takes when updating its parameters during gradient descent.

---

### 5. Dimensions and Sizes
These letters usually represent the sizes of the various vectors and matrices.

*   **$N$ or $m$:** The total number of samples in the entire dataset.
*   **$B$ (or sometimes $b$):** The batch size (number of samples processed at once).
*   **$d$ or $D$:** The dimensionality of the input features (e.g., if $x$ has 10 features, $d = 10$).

---

### A Standard ML Equation in Context
When you see a standard equation like this:

$$\hat{y} = \sigma(Wx + b)$$

You can now translate it as: 
*"The **prediction** ($\hat{y}$) is calculated by taking the **input vector** ($x$), multiplying it by the **weight matrix** ($W$), adding the **bias vector** ($b$), and passing the result through a **non-linear activation function** ($\sigma$)."*


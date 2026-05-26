# Foundation B: Gradients, Loss, and Backprop

This foundation should sit after **Shape Literacy** and before deeper training topics like LoRA, MoE, and optimization.

The goal:

> Understand how a model knows it is wrong, how it knows which direction to change, and how that error signal moves backward through the computation graph.

This connects directly to the linear probe script, where the model predicts labels, computes binary cross entropy loss, uses `value_and_grad`, and updates `w` and `b` through gradient descent. 

---

# 1. The basic training loop

Training is just this loop:

```text
make prediction
↓
measure error
↓
compute direction of improvement
↓
adjust parameters
↓
repeat
```

In ML language:

```text
forward pass
↓
loss
↓
backpropagation
↓
gradient descent update
↓
repeat
```

The model does not “understand” it is wrong.

It gets a number.

That number is the **loss**.

---

# 2. What is loss?

A **loss function** turns model badness into one scalar number.

Example:

```text
loss = 0.01  → very good
loss = 0.50  → okay-ish
loss = 8.00  → bad
```

The loss is the training target.

The model’s job is not directly:

```text
be smart
```

The model’s job is:

```text
make the loss smaller
```

That is the brutally simple center of training.

---

# 3. Example: linear probe prediction

In the linear probe lesson, the model predicts:

```text
z = x · w + b
```

Then:

```text
prediction = sigmoid(z)
```

So for one example:

```text
x = input vector
w = learned weights
b = bias
prediction = probability of Class 1
```

Example:

```text
prediction = 0.90
label = 1.0
```

Good. Low loss.

```text
prediction = 0.10
label = 1.0
```

Bad. High loss.

---

# 4. Loss turns “wrong” into a number

For binary classification, the common loss is **binary cross entropy**.

You do not need to memorize the formula yet. You need the behavior:

```text
confident and correct   → tiny loss
uncertain               → medium loss
confident and wrong     → huge loss
```

Examples:

```text
label = 1, prediction = 0.99 → very low loss
label = 1, prediction = 0.50 → medium loss
label = 1, prediction = 0.01 → very high loss
```

Cross entropy punishes confident wrongness hard.

That is useful because a model should not be confidently wrong.

---

# 5. What is a gradient?

A **gradient** tells you:

> If I slightly change this parameter, will the loss go up or down?

For one parameter, think of a hill:

```text
loss
 ^
 |
 |        *
 |      *
 |    *
 |  *
 |________________> parameter
```

The slope tells you which way the loss is increasing.

The gradient is that slope.

If the gradient is positive:

```text
increasing parameter raises loss
```

So to reduce loss, move the parameter down.

If the gradient is negative:

```text
increasing parameter lowers loss
```

So to reduce loss, move the parameter up.

Core rule:

```text
move opposite the gradient
```

---

# 6. Gradient descent

Gradient descent updates parameters like this:

```text
new_parameter = old_parameter - learning_rate * gradient
```

For weights:

```text
w_new = w_old - lr * grad_w
```

For bias:

```text
b_new = b_old - lr * grad_b
```

The repo’s linear probe script does exactly this: it computes gradients for `w` and `b`, then subtracts `learning_rate * gradient` from each parameter. 

The learning rate controls step size.

Too small:

```text
training is slow
```

Too large:

```text
training jumps around or explodes
```

Just right:

```text
loss steadily decreases
```

---

# 7. What is backpropagation?

Backpropagation is the algorithm that computes gradients through a chain of operations.

Suppose the model does:

```text
x
↓
dot product
↓
add bias
↓
sigmoid
↓
loss
```

Backprop asks:

```text
How much did each earlier thing contribute to the final loss?
```

It works backward:

```text
loss
↑
sigmoid
↑
add bias
↑
dot product
↑
w, b
```

Backprop is just the chain rule applied efficiently.

---

# 8. The chain rule intuition

If:

```text
A affects B
B affects C
C affects loss
```

then A affects loss through the chain:

```text
A → B → C → loss
```

Backprop multiplies those sensitivities backward.

Plain English:

> If changing A changes B, and changing B changes C, and changing C changes loss, then changing A changes loss.

That is the whole idea.

---

# 9. Forward pass vs backward pass

## Forward pass

The model computes outputs.

```text
input → prediction → loss
```

Example:

```text
x → sigmoid(x · w + b) → binary cross entropy
```

## Backward pass

The model computes gradients.

```text
loss → gradients for w and b
```

Then gradient descent updates the parameters.

```text
w = w - lr * grad_w
b = b - lr * grad_b
```

Forward pass produces the mistake.
Backward pass assigns responsibility.

---

# 10. Computation graph

A computation graph is the chain of operations used to compute the loss.

Example:

```text
x ----\
       dot product → z → sigmoid → prediction → loss
w ----/                         label --------/
b ----------------/
```

Every operation knows how its output changes when its inputs change.

Backprop walks this graph backward.

That is why frameworks like Nx, PyTorch, JAX, and TensorFlow can compute gradients automatically.

---

# 11. What `value_and_grad` is doing

In the linear probe script, `value_and_grad` means:

```text
compute the loss value
and also compute gradients of that loss
```

Conceptually:

```text
{loss, gradients} = value_and_grad(parameters, loss_function)
```

So it gives both:

```text
how bad are we?
which direction should parameters move?
```

That is the training engine.

---

# 12. Why gradients are vectors/tensors too

If `w` has shape:

```text
{2, 1}
```

then `grad_w` has the same shape:

```text
{2, 1}
```

Each weight gets its own slope.

If a model has 7 billion parameters, then the gradient has 7 billion corresponding values.

One gradient value per parameter.

So training a neural network is like saying:

```text
for every knob in the model:
  estimate which way to turn it
```

Backprop does this efficiently.

---

# 13. Why loss must be a scalar

Training usually needs one final scalar loss:

```text
loss = 0.347
```

Why?

Because gradients answer:

```text
how does this one number change when each parameter changes?
```

The loss compresses all model errors into one optimization target.

For a batch, the model may make many predictions:

```text
preds shape: {1000, 1}
labels shape: {1000, 1}
```

But the loss becomes:

```text
scalar
```

usually by averaging errors across the batch.

That gives one number to minimize.

---

# 14. Batch training

Instead of updating from one example at a time, models usually train on batches.

Example:

```text
x shape: {batch=1000, features=2}
y shape: {batch=1000, targets=1}
```

The model predicts:

```text
preds shape: {1000, 1}
```

Then computes average loss:

```text
loss = mean(error over 1000 examples)
```

This makes the gradient more stable.

A single example may be noisy.
A batch gives a better estimate of the direction to improve.

---

# 15. Why softmax collapse killed gradients

This connects to the previous lesson.

Softmax collapse means the output becomes almost:

```text
[0, 0, 1, 0]
```

The Jacobian becomes almost zero.

That means:

```text
changing the input scores barely changes the output probabilities
```

So backprop receives almost no useful signal.

In training language:

> The gradient cannot flow backward through collapsed softmax.

That is why scaling attention scores matters.

It keeps the computation graph sensitive.

---

# 16. Why regularization appears in the loss

The linear probe script adds L2 weight decay.

That means the loss is not only:

```text
classification error
```

It is:

```text
classification error + penalty for large weights
```

So the model is trained to be:

```text
accurate
and
simple
```

This discourages huge fragile weights that separate the training data but may generalize poorly.

Regularization changes the optimization target.

The model is no longer just minimizing wrongness.
It is minimizing wrongness plus complexity.

---

# 17. Training vs inference

This distinction is essential.

## Training

```text
predictions are made
loss is computed
gradients are computed
weights are updated
```

## Inference

```text
predictions are made
no loss required
no gradients required
weights stay fixed
```

During inference, the model is just running forward.

During training, the model runs forward and backward.

---

# 18. Why LoRA training is cheaper

This foundation also explains LoRA.

Full fine-tuning computes gradients for many original weights.

LoRA freezes the original weights and only computes updates for the small low-rank matrices.

So:

```text
W₀ is frozen
A and B are trainable
```

Backprop still happens, but fewer parameters receive gradients.

That means:

```text
less memory
less compute
fewer trainable knobs
```

LoRA is not magic. It is gradient descent on a smaller parameter set.

---

# 19. Why gradients are local, but learning looks global

A gradient only tells you the best immediate small step.

It does not know the entire path to the best solution.

So training is iterative:

```text
small step
small step
small step
small step
...
```

Each update changes the landscape slightly because the parameters changed.

Training is not one perfect move.

It is many local corrections.

---

# 20. Common failure modes

## Learning rate too high

```text
loss jumps around
training unstable
parameters overshoot
```

## Learning rate too low

```text
loss decreases very slowly
training takes forever
```

## Bad loss function

```text
model optimizes the wrong behavior
```

## Dead gradients

```text
parameters stop receiving useful update signal
```

## Overfitting

```text
training loss gets low
validation performance stays bad
```

## Underfitting

```text
model cannot reduce training loss enough
```

---

# 21. The core mental model

Training is not “the model thinking.”

Training is:

```text
1. make output
2. score output with loss
3. compute slopes of loss with respect to parameters
4. move parameters opposite those slopes
5. repeat
```

Backprop is the mechanism that computes those slopes efficiently.

Gradient descent is the mechanism that uses those slopes to update parameters.

Loss is the scoreboard.

---

# 22. Practice questions

Answer these before continuing:

1. What is a loss function?
2. Why does training need the loss to be a scalar?
3. What does a gradient tell you?
4. Why do we move opposite the gradient?
5. What does the learning rate control?
6. What is the difference between forward pass and backward pass?
7. What does backpropagation compute?
8. Why do gradients have the same shape as the parameters?
9. Why does softmax collapse hurt backprop?
10. What is the difference between training and inference?
11. Why does L2 regularization change the loss?
12. Why is LoRA cheaper from a gradient perspective?

---

# One-sentence summary

**Loss tells the model how wrong it is, gradients tell each parameter which way to move, and backprop efficiently carries that error signal backward through the computation graph.**

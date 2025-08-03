---
author: Saurav
date: '2023-06-18'
title: Log Sum Exp in Pytorch 
# subtitle: A CSS library for creating beautiful Tufte-inspired HTML documents.
meta: true
math: true
hasMath: true
toc: false
draft: false
hideDate: false
hideReadTime: true
categories: [misc]
description: "math tricks for stability in Pytorch"
---

### Why PyTorch Combines Softmax and Loss: A Look at Numerical Stability

When working with PyTorch for classification tasks, you'll encounter a common pattern: your model outputs raw scores (called **logits**), and you feed these directly into a loss function like `nn.CrossEntropyLoss` without first converting them to probabilities.

This might seem counterintuitive. Why not take the logical step of applying the softmax function to get probabilities first? This article explores the question and explains the clever reason behind PyTorch's design.

---

### The Question: Why Not Apply Softmax Before Calculating Loss?

On the surface, a two-step process for calculating the loss for a classification model seems correct:

1.  **Get Probabilities:** Take the model's raw logit scores, \\(z\\), and apply the softmax function to convert them into a probability distribution. The formula for the probability of class \\(i\\) is \\(p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}\\).
2.  **Calculate Loss:** Use the Negative Log-Likelihood (NLL) to find the loss, which is the negative logarithm of the predicted probability for the correct class, \\(y\\). The formula is \\(L = -\log(p_y)\\).

This approach is mathematically sound, but when implemented on a computer, it runs into significant problems related to numerical stability.

---

### The Problem: Numerical Instability

The issue lies with the exponential function (\\(e^z\\)) inside the softmax formula. Computers can only represent numbers within a finite range, and the exponential function can easily produce numbers that fall outside this range.

**Overflow Risk:** If a logit is a large positive number (e.g., \\(z_i = 100\\)), its exponential, \\(e^{100}\\), is an astronomically large number. This exceeds the maximum value a standard floating-point number can hold, leading to an **overflow**. The result becomes `infinity` or `NaN` (Not a Number), which breaks the entire training process as the gradients become undefined.

**Underflow Risk:** If a logit is a large negative number (e.g., \\(z_i = -1000\\)), its exponential, \\(e^{-1000}\\), is a number infinitesimally close to zero. The computer may round this down to exactly `0.0`. When you then try to compute the loss, you are faced with calculating \\(-\log(0)\\), which is `infinity`. Again, the loss explodes, and training fails.

Because backpropagation relies on a well-defined loss value to compute gradients, these `infinity` or `NaN` results stop learning in its tracks.

---

### The Solution: A Combined, Stable Calculation

To avoid these pitfalls, PyTorch's `nn.CrossEntropyLoss` combines the softmax and NLL calculations into a single, mathematically equivalent but far more stable operation.

The derivation starts by substituting the softmax formula directly into the NLL loss function:
$$L = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)$$
Using the logarithm property $\log(\frac{a}{b}) = \log(a) - \log(b)$, we can rewrite this as:
$$L = -(\log(e^{z_y}) - \log(\sum_j e^{z_j}))$$
Since $\log(e^{z_y}) = z_y$, this simplifies to:
$$L = -z_y + \log\left(\sum_j e^{z_j}\right)$$
This new expression still contains the term $\log(\sum_j e^{z_j})$, known as the **Log-Sum-Exp (LSE)** function, which could still overflow. This is where the final "trick" comes in. We can find the maximum logit value, $m = \max(z)$, and rewrite the LSE term without changing its value:
$$L = -z_y + m + \log\left(\sum_j e^{z_j - m}\right)$$
This final formula is the key. By subtracting the maximum logit $m$ from each logit $z_j$, the exponent $(z_j - m)$ is **always less than or equal to zero**. This prevents the exponential term from ever becoming a huge number, thus avoiding overflow. This stable calculation is what `nn.CrossEntropyLoss` performs under the hood.

---

### The Role of Softmax for Inference

While this combined function is crucial for stable **training**, the softmax function is still necessary during **inference** (when making predictions). The goal of inference is to get an interpretable output. The raw logits from the model are just scores; they are not probabilities.

To convert these scores into a probability distribution that sums to one, you must explicitly call `torch.softmax()` on the model's output.

```python
# During training, we feed raw logits to the loss function
# loss = nn.CrossEntropyLoss(logits, labels)

# During inference, we apply softmax to get probabilities
with torch.no_grad():
    logits = model(X)
    probabilities = torch.softmax(logits, dim=1)

print("Logits:", logits)
print("Probabilities:", probabilities)
```

Note : `no_grad` is an optimisation in Pytorch that allows users to let go of the gradients while inference to make it more memory efficient. 

The standalone `torch.softmax` function is also implemented to be numerically stable(with the same LogSumExp trick), so it's safe to call for prediction.

### Summary

- Separating the softmax and NLL loss calculations can lead to numerical overflow or underflow during training, causing the loss to become `infinity` or `NaN`.
- PyTorch's `nn.CrossEntropyLoss` combines these two steps into a single, numerically stable operation using the **Log-Sum-Exp trick**.
- For this reason, you should always feed **raw logits** directly to the loss function during training.
- During **inference**, you should apply `torch.softmax()` to the logits to get a final, interpretable probability distribution, which also applies the same same trick as LogSumExp. 

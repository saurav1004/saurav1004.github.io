---
author: Saurav
date: '2023-07-02'
title: Numerical Stability in Pytorch
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

#### -- old post -- 
#### -- maintainence notice : I wrote this post long back, and have moved multiple templates / static site generators in the meantime. Most expressions should just work fine. But it might be breaking in some places, thanks for bearing with it. -- 

The core of training a deep neural network is gradient-based optimization. We calculate how the model's error changes with respect to each of its millions (or billions) of parameters and take a small step in the direction that reduces that error. While simple in theory, this process is fraught with numerical challenges. The gradients themselves, the very signals that guide learning, can become unruly.

In this post, we will explore two fundamental techniques that every deep learning practitioner should understand, both designed to control the magnitude of gradients. First, we'll look at **Loss Scaling**, which rescues gradients that become too small. Second, we'll examine **Gradient Clipping**, which tames gradients that become too large. Together, they form a crucial part of the toolkit for stable and efficient model training.

---

## Part 1: Rescuing Small Gradients with Loss Scaling

Training large-scale models is a task of immense computational and memory cost. To make this feasible, deep learning frameworks have widely adopted **mixed-precision training**, which leverages lower-precision numerical formats to accelerate computation and reduce memory usage. However, this efficiency comes with a numerical stability challenge that requires a clever solution.

### The Challenge of Gradient Underflow

In standard training, most variables are stored as 32-bit floating-point numbers (`float32`). Mixed-precision training, by contrast, performs many operations using 16-bit floating-point numbers (`float16`), which halves memory usage and dramatically speeds up calculations on modern GPUs.

The critical trade-off is the reduced dynamic range of `float16`. In deep networks, as gradients are propagated backward, their magnitudes can become exceedingly small. Consider a gradient with a magnitude of \\(6 \times 10^{-8}\\). While perfectly representable in `float32`, this value is smaller than the smallest positive number `float16` can represent (approx. \\(6.1 \times 10^{-5}\\)). When cast to `float16`, this gradient is rounded to zero.

This phenomenon is known as **gradient underflow**. When a gradient is flushed to zero, the corresponding weights are not updated. The learning signal is lost, and model convergence can stall.

### The Elegant Solution: Loss Scaling

The solution is a numerically simple yet powerful technique called **loss scaling**. Before initiating the backward pass, we simply multiply the calculated loss value, \\(L\\), by a large scaling factor, \\(S\\).

$$L_{scaled} = L \cdot S$$

By the chain rule, this scaling factor propagates through the entire backward pass. The gradient of the scaled loss with respect to any parameter \\(w\\) is therefore the original gradient, also scaled by \\(S\\):

$$\frac{\partial L_{scaled}}{\partial w} = \frac{\partial (L \cdot S)}{\partial w} = S \cdot \frac{\partial L}{\partial w}$$

This multiplication shifts the gradient values upwards. Our tiny gradient of \\(6 \times 10^{-8}\\), when scaled by \\(S=32768\\), becomes approximately \\(1.97 \times 10^{-3}\\), which is well within the `float16` range.

Before the optimizer updates the weights, the scaled gradients are unscaled by dividing by \\(S\\), restoring their original values, now free from underflow.

$$\nabla_w L = \frac{\nabla_w L_{scaled}}{S}$$

Frameworks like PyTorch automate this with **dynamic loss scaling** (`torch.cuda.amp.GradScaler`), which adjusts \\(S\\) during training—decreasing it if gradients overflow to infinity, and increasing it during stable periods to capture even smaller gradients.

---

## Part 2: Taming Large Gradients with Gradient Clipping

On the opposite end of the spectrum from vanishing gradients is the problem of **exploding gradients**. This occurs when the gradient of the loss function grows excessively large, leading to unstable training.

### The Mathematical Reason for Exploding Gradients

The weight update rule at the heart of training is:

$$\theta_{new} = \theta_{old} - \eta \cdot \nabla_{\theta}L$$

Here, \\(\eta\\) is the learning rate and \\(\nabla_{\theta}L\\) is the gradient. The problem of exploding gradients occurs when the norm of the gradient, \\(||\nabla_{\theta}L||\\), becomes excessively large.

By the chain rule, the gradient in an early layer is a product of many Jacobian matrices from all subsequent layers. For a loss \\(L\\) and the weights \\(W_l\\) of layer \\(l\\), the gradient is computed backwards from the last layer (\\(n\\)):

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdot \frac{\partial a_{n-1}}{\partial a_{n-2}} \cdots \frac{\partial a_{l+1}}{\partial a_l} \cdot \frac{\partial a_l}{\partial W_l}$$

If the norms of the Jacobian terms \\(\frac{\partial a_{k+1}}{\partial a_k}\\) are consistently greater than 1, their product can grow exponentially. The resulting gradient vector can have an enormous magnitude, causing an update so large that it catapults the weights into a poor region of the loss landscape or results in floating-point overflow (`NaN`). This is especially common in Recurrent Neural Networks (RNNs) due to the repeated application of the same weight matrix.

### The Solution: The Mathematics of Gradient Clipping

Gradient Clipping directly addresses this by imposing a ceiling on the magnitude of the gradient vector. It is applied *after* the gradients are calculated but *before* the optimizer updates the weights.

**Step 1: Define the Global Gradient Vector**
First, we aggregate the gradients for all trainable parameters ($\theta$) of the model into a single, large vector, which we'll call \\(g\\).

$$g = \nabla_{\theta}L$$

**Step 2: Compute the Norm of the Gradient Vector**
Next, we compute the L2 norm (Euclidean norm) of this vector \\(g\\). The norm is a scalar value representing the overall magnitude or "length" of the gradient vector.

$$||g||_2 = \sqrt{g_1^2 + g_2^2 + \dots + g_N^2}$$

**Step 3: The Clipping Logic**
We define a hyperparameter, a scalar value called `max_norm`, which is the maximum permissible norm for our gradient vector. We then apply the following condition:

-   If \\(||g||_2 \le \text{max\_norm}\\), the gradient's magnitude is acceptable, and we do nothing.
-   If \\(||g||_2 > \text{max\_norm}\\), the gradient has "exploded," and we must rescale it. The clipped gradient, \\(\hat{g}\\), is calculated as follows:

$$\hat{g} = \left( \frac{\text{max\_norm}}{||g||_2} \right) \cdot g$$

Since we are in the case where \\(||g||_2 > \text{max\_norm}\\), the scalar term in the parenthesis is less than 1. Multiplying the original gradient vector \\(g\\) by this scalar shrinks its length without changing its direction. The new, rescaled gradient vector \\(\hat{g}\\) will have a norm exactly equal to `max_norm`.

**Step 4: Update the Weights with the Clipped Gradient**
Finally, the optimizer uses this potentially rescaled gradient \\(\hat{g}\\) to perform the weight update:

$$\theta_{new} = \theta_{old} - \eta \cdot \hat{g}$$

This ensures that even if the original gradient was enormous, the update step is limited to a reasonable size, keeping the training process stable.

## Conclusion

While they address opposite problems, loss scaling and gradient clipping are two sides of the same coin: **gradient magnitude control**. Successful deep learning depends on keeping the gradient signal within a "Goldilocks zone"—not so small that it's lost to numerical precision (underflow), and not so large that it destabilizes the entire training process (overflow).

Loss scaling pushes gradients up from the floor of `float16` representation, while gradient clipping pulls them down from the ceiling of numerical instability. They are simple, effective, and indispensable tools for training robust and powerful deep learning models.
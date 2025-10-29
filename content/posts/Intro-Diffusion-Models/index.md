---
title: "Diffusion Models: Theory and Applications in Earth Sciences"
summary: "A beginner’s guide to diffusion models"
date: 2025-10-28
tags: ["Diffusion Model"]
author: "ESS-AIML"
series: ["AI-ML"]
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: false
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
math: true

cover:
    image: "images/Earth_diffusion.jpg" # image path/url
    alt: "<alt text>" # alt text
    caption: "Photo created by Text-to-Image AI" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-aiml/blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

In the past few years, **diffusion models** have become one of the most powerful tools in artificial intelligence (AI). They’re the engines behind many of today's most impressive **generative systems** — from creating realistic images, sounds, and videos to helping design new molecules and medicines to modeling complex climate and environmental systems.

There are already plenty of great articles that dive into the details of diffusion models — and we’ll share some of our favorites along the way. In this post, we'll keep thing simple: ***we focus on the core principles and explore how diffusion models are being used in environmental system science***, and why those applications are so promising and exciting.

Let’s get started!

## What are generative models?

> *Generative models are a type of AI systems that can learn the underlying structure of existing data and use it to create new content similar to the originals.*

What does this mean in practice? Suppose we have a dataset containing photos of dogs.
A generative model can study all those images to learn what makes a picture look like a dog - the shapes, colors, textures, and relationships between pixels. Once trained, the model can then generate completely new, realistic images of dogs that did not exist in the original dataset.

Generative models are also **probabilistic**, meaning they don’t always produce the same output. Instead, they can create many different versions of an image or dataset, all slightly varied but still realistic. This makes them especially useful for creative tasks, predictive simulation, and risk-based scientific modeling.


{{< figure
  src="images/generative_modeling.jpg"
  alt="Generative"
  caption="Figure 1: Generative model learns features of dogs from the training dataset and can generate new, high-quality dog images. Credit: Tanishq Mathew Abraham."
>}}

There are different types of generative models, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), flow-based models, and diffusion models. Each type has its strengths and weaknesses, but diffusion models have recently shown outstanding performance in producing high-quality, realistic results. We’ll focus on diffusion models in this post.


{{< figure
  src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"
  alt="Generative"
  caption="Figure 2: Overview of different types of generative models. Source: [Lil'Log](https://lilianweng.github.io/)"
>}}


## What are diffusion models?
Diffusion models are inspired by non-equilibrium thermodynamics -- specifically, how particles spread out or "diffuse" over time. ***The core idea behind them is simple:*** we gradually corrupt (add noise to) clean data until it becomes completely random, then train a deep learning model to reverse this process and recover the original data.

> *Diffusion models are a class of generative models that learn to reverse a gradual noising process applied to data, enabling them to generate realistic samples from the underlying data distributions by iteratively denoising random noise.*

In other words, diffusion models learn how to "undo" noise. Imagine taking a blurry or noisy satellite image and carefully sharpening it, one small step at a time, until continents and clouds slowly come back into focus. Each step removes a bit of noise, turning random patterns into something meaningful.

In principle, if we start from pure random noise, we should be able to keep applying the trained model until we obtain a sample that looks as if it were drawn from the training set. That's it -- and yet this simple idea works incredibly well in practice.

*For a more visual and intuitive explanation, check out **[this article](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models)** -- it provides an interactive, step-by-step introduction that makes diffusion models much easier to grasp.*

## How do diffusion models work?
Now, let’s explore how diffusion models actually work. The process has two main stages: a **forward diffusion process** -- where noise is gradually added to data until it becomes completely random, and a **reverse diffusion process** -- where the model learns to remove that noise step by step to reconstruct the original data.

{{< figure
  src="images/diffusion_processes.jpg"
  alt="Diffusion model"
  caption="Figure 3: Forward and reverse diffusion processes."
>}}

### Forward diffusion
Suppose we have a real data sample $\mathbf{x}_0 \sim q(\mathbf{x})$. In the forward process, we gradually corrupt the sample by adding small amounts of Gaussian noise over $T$ steps, producing a sequence of increasingly noisy samples $\mathbf{x}_1, \dots, \mathbf{x}_T$.
The amount of noise added at each step $t$ is controlled by a variance schedule $\\{\beta\_t \in (0, 1)\\}\_{t=1}^T$.
$$
\begin{aligned}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) &= \mathcal{N}(\mathbf{x}\_t; \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}, \beta\_t\mathbf{I}) \\\
q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0) &= \prod^T\_{t=1} q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1})
\end{aligned}
$$

As $t$ increases, the sample $\mathbf{x}_t$ becomes progressively noisier.
Eventually when $T \rightarrow \infty$, $\mathbf{x}_T$ is indistinguishable from random noise.
Mathematically, we can write each step of this process as follows:
$$
\mathbf{x}\_t = \sqrt{1-\beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1}  \quad \quad \text{where }\boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Here, $\mathcal{N}(\cdot,\cdot)$ denotes a normal distribution.
Since the sum of two Gaussian variables with variances $\sigma^2_1$ and $\sigma^2_2$ is also Gaussian with variance $\sigma^2_1+\sigma^2_2.$ Given that $\boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, if $\mathbf{x}\_{t-1}$ has zero mean and unit variance, then so does $\mathbf{x}_{t}$, as $\sqrt{1-\beta\_t}^2 + \sqrt{\beta\_t}^2=1$.
This scaling ensures that the variance remains stable throughout the diffusion process.
This way, if we normalize our original sample $\mathbf{x}\_{0}$ to have zero mean and unit variance, then the sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ will also maintain these properties and $\mathbf{x}_T$ will approximate a standard Gaussian distribution for sufficiently large $T$.

Another nice property of the above process is that we can jump straight from the original sample $\mathbf{x}_0$ to any noised version of the forward diffusion process $\mathbf{x}_t$ using a clever reparameterization trick as below.

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$, then we can write the following:
$$
\begin{aligned}
\mathbf{x}\_t
&= \sqrt{\alpha\_t}\mathbf{x}\_{t-1} + \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon}\_{t-1} \\\
&= {\color{blue}\sqrt{\alpha\_t}\( \sqrt{\alpha\_{t-1}}\mathbf{x}\_{t-2} + \sqrt{1 - \alpha\_{t-1}}\boldsymbol{\epsilon}\_{t-2} \)} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
&= {\color{blue}\sqrt{\alpha\_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + \sqrt{\alpha\_t (1-\alpha\_{t-1})}\boldsymbol{\epsilon}\_{t-2}} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
&= \sqrt{\alpha\_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + {\color{blue}\sqrt{1 - \alpha\_t \alpha\_{t-1}} \bar{\boldsymbol{\epsilon}}\_{t-2} } \\\
&= \dots \\\
&= \sqrt{\bar{\alpha}\_t}\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\boldsymbol{\epsilon}
\end{aligned}
$$

Since $\boldsymbol{\epsilon}\_{t-2}, \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, their weighted sum is itself a Gaussian with standard deviation $\sqrt{\alpha_t (1-\alpha_{t-1})+(1-\alpha_t)} = \sqrt{1-\alpha_t\alpha_{t-1}}$ and $\bar{\boldsymbol{\epsilon}}\_{t-2} \sim \mathcal{N}(\mathbf{0})$.

The forward diffusion process $q$ can therefore be written as follows:
$$
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

### Reverse diffusion
The reverse diffusion process does the opposite of the forward one: *it learns how to remove the added noise, step by step, to gradually recover the original data*.
In theory, if we can *undo* the forward process and sample from $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$, we can reconstruct a true data sample starting from pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

In practice, computing the conditional probability $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ is however **intractable**, because it would require integrating over the entire data distribution.
So instead, we train a neural network $p_\theta$ to approximate this reverse process.
The model learns to predict what the "cleaner" version of the data should look like at each step -- essentially learning how to reverse the noising process, one small step at a time.

{{< figure
  src="images/reverse_process.jpg"
  alt="Diffusion model"
  caption="Figure 4: Reverse diffusion process."
>}}

Because each transition in the forward process adds only a tiny bit of Gaussian noise, the reverse transitions can also be modeled as Gaussian distributions:

$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) = \mathcal{N}(\mathbf{x}\_{t-1}; \boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t), \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t))
$$

If we apply the reverse formula for all timesteps from $t=T \rightarrow 0$, we can go from random noise $\mathbf{x}\_T$ to a coherent, realistic sample that is similar to $\mathbf{x}\_0$:
$$
p\_\theta(\mathbf{x}\_{0:T}) = p(\mathbf{x}\_T) \prod^T\_{t=1} p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)
$$

### How do diffusion models learn
Training a diffusion model means teaching it how to predict and remove noise correctly.
During training, we know both the original data $\mathbf{x}\_0$ and its noisy version $\mathbf{x}\_t$ created by the forward process. ***The model’s goal is simple: predict the exact noise that was added***. The more accurately it can do this, the better it becomes at reversing the diffusion and generating realistic samples.

To guide this learning, we use a loss function -- essentially it measures how far the model’s predicted noise is from the actual noise that was added. The most common loss function looks like this:

$$
L = \mathbb{E}\_{\mathbf{x}\_0, \epsilon, t} \left[ \| \epsilon - \epsilon\_\theta(\mathbf{x}\_t, t) \|^2 \right]
$$
where $\mathbf{x}\_0$ is the original clean data, $\epsilon$ is the true Gaussian noise added at time step $t$, $\epsilon_\theta(\mathbf{x}\_t, t)$ is the noise predicted by the model, and $\mathbf{x}\_t$ is the noised version of the data at step $t$.
We minimize the loss function by tuning Gaussian parameters (meaning the mean $\boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t)$ and the covariance matrix $\boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t)$) for each timestep.
This way, the model learns how to progressively remove noise from any noisy input — allowing it to generate realistic samples starting from pure random noise.

Behind the scenes, this loss comes from something called the Variational Lower Bound (VLB) — a statistical framework that connects the model’s noise predictions to the overall likelihood of generating real data.
You can think of it as a mathematical way of ensuring the model learns the most likely way to transform random noise into meaningful patterns.

In this post, we skip the complicated derivation and use the simplified loss above, since it captures the same idea in a much more efficient way.
If you’d like to understand the full math, check out [**Lil'Log's post**](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) or [**The AI Summer’s guide**](https://theaisummer.com/diffusion-models/). Both resources give detailed explanations of the math and intuition behind it.

**So in short:**
- We add noise to data (the forward process).
- The model learns to remove that noise (the reverse process).
- We train it by minimizing the difference between actual noise and predicted noise.

---
### Diffusion models in environmental system science 🌍
The Earth’s environment is an intricate web of interacting systems. From regular daily temperature shifts to extremes events like droughts, hailstorms, and the El Niño–Southern Oscillation (ENSO), natural variability drives much of the planet’s behavior. These fluctuations shape water availability, agricultural yields, transportation safety, and even energy production. Accurate and timely forecasts of such variability are therefore essential — enabling societies to prepare for hazards, optimize renewable energy use, and adapt to a changing climate.

In recent years, diffusion models have opened a new path for data-driven Earth and environmental system simulations, one that differs sharply from both classical numerical models and earlier neural networks such as GANs or VAEs. Rather than mapping inputs to outputs in a single deterministic step, diffusion models generate spatio-temporal fields by progressively refining random noise—each iteration learning how real environmental patterns emerge from stochastic variability.

{{< figure
  src="images/Guilloteau2024.gif"
  alt="Diffusion model"
  caption="Diffusion-based Ensemble Rainfall estimation from Satellite. Source [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)"
>}}

[Guilloteau et al. (2025)](https://ieeexplore.ieee.org/abstract/document/10912662) present a generative diffusion model for producing probabilistic ensembles of precipitation maps conditioned on multisensor satellite observations. Their framework integrates physical and statistical reasoning, using diffusion-based generative learning to reconstruct fine-scale rainfall structures from coarse satellite inputs. Unlike traditional deterministic downscaling or regression models, the diffusion approach captures spatial uncertainty and variability inherent in precipitation processes, resulting in physically consistent rainfall ensembles. This method demonstrates the potential of diffusion models to enhance probabilistic rainfall estimation and satellite-based hydrological prediction.

Because diffusion models explicitly model the distribution of states rather than just the mean response, they are particularly well-suited for capturing uncertainty, extremes, and multi-scale variability—features that are notoriously difficult for traditional deep learning architectures. For instance, recent studies [[Bassetti et al, 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194), [Hess et al, 2025](https://www.nature.com/articles/s42256-025-00980-5)] have shown that diffusion-based emulators can reconstruct fine-scale rainfall structures from coarse reanalysis data while preserving the physical coherence of storm systems, something most conventional downscaling models tend to blur.

**To be continue...**

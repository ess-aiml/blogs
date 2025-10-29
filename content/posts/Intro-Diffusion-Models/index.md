---
title: "Diffusion Models: Principles and Applications in Earth Sciences"
summary: "A beginner’s guide to diffusion models"
date: 2025-10-28
tags: ["Diffusion Model", "Generative", "Earth system"]
author: "Phong Le"
series: ["AI-ML"]
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: true
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
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
    caption: "Photo created by AI" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-aiml/blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

Diffusion models have become one of the most powerful tools in artificial intelligence (AI). They’re the engines behind some of today's most advanced ***generative systems*** -- from creating realistic images, audio, text, and videos to designing new molecules and medicines, and even modeling complex climate and environmental systems.

There are already plenty of great articles that dive into the details of diffusion models -- and we’ll share some of our favorites along the way. In this post, we'll keep thing simple: we focus on the core principles and explore how diffusion models are being used in environmental science, and why those applications are so promising and exciting.

Let’s get started!

## What are generative models?

> *Generative models are a type of AI systems that learn the underlying structure of existing data and use it to create new content that resembles to the original.*

What does this mean in practice? Suppose we have a dataset containing photos of dogs.
A generative model can study all those images to learn what makes a picture look like a dog - the shapes, colors, textures, and relationships between pixels. Once trained, the model can then generate completely new, realistic images of dogs that did not exist in the original dataset.

Generative models are also **probabilistic**, meaning they don’t always produce the same output. Instead, they can create many different versions of an image or dataset, all slightly varied but still realistic. This makes them especially useful for creative tasks, predictive simulation, and risk-based scientific modeling.


{{< figure
  src="images/generative_modeling.jpg"
  alt="Generative"
  caption="Figure 1: A generative model learns features from the training dataset and can generate new, high-quality images. Source: Photo from [Tanishq Mathew Abraham](https://x.com/iscienceluvr/status/1592860024657051649)."
>}}

There are different types of generative models, such as [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661) (GANs), [Variational Autoencoders](https://arxiv.org/abs/1312.6114) (VAEs), [flow-based models](https://en.wikipedia.org/wiki/Flow-based_generative_model), and [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model). Each type has its strengths and weaknesses, but diffusion models have recently shown outstanding performance in producing high-quality, realistic results. We’ll focus on diffusion models in this post.


{{< figure
  src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"
  alt="Generative"
  caption="Figure 2: Computation graphs of prominent generative models. Source: [Lil'Log](https://lilianweng.github.io/)"
>}}


## What are diffusion models?
Diffusion models are inspired by non-equilibrium thermodynamics -- specifically, how particles spread out or "diffuse" over time. The core idea behind them is simple: we gradually corrupt (add noise to) clean data until it becomes completely random, then train a deep learning model to reverse this process and recover the original data.

> *Diffusion models are a class of generative models that learn to reverse a gradual noising process applied to data, enabling them to generate realistic samples from the underlying data distributions by iteratively denoising random noise.*

In other words, diffusion models learn how to "undo" noise. Imagine taking a blurry or noisy satellite image and carefully sharpening it, one small step at a time, until continents and clouds slowly come back into focus. Each step removes a bit of noise, turning random patterns into something meaningful.

In principle, if we start from pure random noise, we should be able to keep applying the trained model until we obtain a sample that looks as if it were drawn from the training set. That's it -- and yet this simple idea works incredibly well in practice.

>*For a more intuitive explanation, check out [this article](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models) -- it provides an interactive, step-by-step introduction that makes diffusion models much easier to grasp.*

Diffusion models come in different forms, depending on how they add and remove noise -- some are probabilistic, while others are deterministic.
One of the most important and widely used approaches is the [Denoising Diffusion Probabilistic Model](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) (DDPM), which has become the basis for many breakthroughs in generative AI.

## How do diffusion models work?
Now, let’s explore how DDPMs actually work. At their core, DDPMs involve two distinct stochastic processes: a ***forward diffusion pass*** -- where noise is gradually added to data until it becomes purely random, and a ***reverse denoising process*** -- where the model learns to remove that noise step by step to reconstruct the original data.

{{< figure
  src="images/diffusion_processes.jpg"
  alt="Diffusion model"
  caption="Figure 3: Forward and reverse diffusion processes."
>}}

### Forward process
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

Since $\boldsymbol{\epsilon}\_{t-2}, \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, their weighted sum is itself a Gaussian with standard deviation $\sqrt{\alpha_t (1-\alpha_{t-1})+(1-\alpha_t)} = \sqrt{1-\alpha_t\alpha_{t-1}}$ and $\bar{\boldsymbol{\epsilon}}\_{t-2} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$.

The forward diffusion process $q$ can therefore be written as follows:
$$
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

### Reverse denosing process
The reverse diffusion process does the opposite of the forward one: *it learns how to remove the added noise, step by step, to gradually recover the original data*.
In theory, if we can *undo* the forward process and sample from $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$, we can reconstruct a true data sample starting from pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

In practice, computing the conditional probability $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ is **intractable**, particularly given the complexity of the data distribution, because it would require integrating over the entire data space.

So instead, we train a learnable parametric model $p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ to approximate the unknown true reverse transition kernel $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$.
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

Behind the scenes, this loss comes from something called the **Variational Lower Bound (VLB)** — a statistical framework that connects the model’s noise predictions to the overall likelihood of generating real data.
You can think of it as a mathematical way of ensuring the model learns the most likely way to transform random noise into meaningful patterns.

In this post, we skip the complicated derivation and use the simplified loss above, since it captures the same idea in a much more efficient way.
If you’d like to understand the full math, check out [Lil'Log's post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), [The AI Summer’s guide](https://theaisummer.com/diffusion-models/), and [this monograph](https://www.arxiv.org/pdf/2510.21890) . All resources give detailed explanations of the math and intuition behind it.

**So in summary:**
- We gradually add noise to data (the forward process).
- The model learns to remove that noise (the reverse process).
- The model aims to minimize the difference between actual noise and predicted noise.

---
### Diffusion models in environmental system science 🌍
Earth’s environment is a complex network of interacting systems. From everyday temperature changes to extreme events like droughts, hailstorms, and El Niño, natural variability shapes nearly every aspect of our planet’s behavior. These fluctuations influence water supply, agriculture, transportation, and energy production. Accurate forecasts of such variability are crucial—they help societies prepare for hazards, manage renewable resources efficiently, and adapt to a changing climate.

Diffusion models offer a fresh approach to simulating Earth and environmental systems. Unlike traditional numerical models or earlier neural networks such as GANs and VAEs, diffusion models don’t rely on a single, fixed mapping from inputs to outputs. Instead, they start with random noise and gradually refine it over many steps, learning how realistic environmental patterns emerge from underlying randomness.

{{< figure
  src="images/Guilloteau2024.gif"
  alt="Diffusion model"
  caption="Diffusion-based Ensemble Rainfall estimation from Satellite (DifERS). Source [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)"
>}}

An example comes from [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662), who developed a generative diffusion framework called DifERS for producing ensembles of precipitation maps from multisensor satellite data. Their method combines physical insight with statistical learning to reconstruct detailed rainfall patterns from coarse satellite inputs. Two novelties of their method thus are: 1) the handling of the uncertainty through the generation of ensembles of equiprobable realizations and 2) the use of coincident measurements from different instruments and different platforms.

Because diffusion models explicitly model the distribution of states rather than just the mean response, they are particularly well-suited for capturing uncertainty, extremes, and multi-scale variability—features that are notoriously difficult for traditional deep learning architectures. For instance, recent studies [[Bassetti et al, (2024)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194), [Hess et al, (2025)](https://www.nature.com/articles/s42256-025-00980-5)] have shown that diffusion-based emulators can reconstruct fine-scale rainfall structures from coarse reanalysis data while preserving the physical coherence of storm systems, something most conventional downscaling models tend to blur.

**To be continue...**

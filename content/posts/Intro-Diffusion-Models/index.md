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

In the past few years, **diffusion models** have become one of the most powerful tools in artificial intelligence (AI). They’re behind many impressive **generative systems** — from creating realistic images, sounds, and videos to helping design new molecules and drugs to modeling complex climate and environmental systems.

There are already plenty of great articles that dive into the technical details of diffusion models — and we’ll share some of our favorites along the way. In this post, we focus on the core principles and explore how diffusion models are being used in environmental system science, and why those applications are so promising and exciting.

Let’s get started!

## What are generative models?

> *Generative models are AI systems that can learn the underlying structure of existing data and use it to create new content similar to the originals.*

What does this mean in practice? Suppose we have a dataset containing photos of dogs. We can train a generative model on this dataset to capture the rules that govern the complex relationships between pixels in images of dogs. Then we can sample from this model to create novel, realistic images of dogs that did not exist in the original dataset.
Generative models are also **probabilistic systems** -- that means they are able to sample many different variations of the output, rather than get the same output every time.

{{< figure
  src="images/generative_modeling.jpg"
  alt="Generative"
  caption="Figure 1: Generative model learns features of dogs from the training dataset and can generate new, high-quality dog images. Credit: Tanishq Mathew Abraham."
>}}

There are different types of generative models, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and flow-based and diffusion models. They have shown great success in generating high-quality samples, but each has some limitations of its own. We only discuss diffusion models here.


{{< figure
  src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"
  alt="Generative"
  caption="Figure 2: Overview of different types of generative models."
>}}


## What are diffusion models?
Diffusion models are inspired by non-equilibrium thermodynamics in physics. The core idea behind them is simple: we gradually corrupt (add noise to) clean data until it becomes completely random, then train a deep learning model to reverse this process and recover the original data.

> *Diffusion models are a class of generative models that learn to reverse a gradual noising process applied to data, enabling them to generate realistic samples from the underlying data distributions by iteratively denoising random noise.*

In other words, diffusion models learn how to "undo" noise — like taking a blurry satellite image and sharpening it bit by bit until the continents and clouds come back into focus. Each small step improves the image, gradually turning random noise into something realistic.
This process has two parts:
- **Forward diffusion**: adding noise to the data in many small steps.
- **Reverse diffusion**: training the model to remove the noise and rebuild the data.

{{< figure
  src="images/diffusion_processes.jpg"
  alt="Diffusion model"
  caption="Figure 3: Forward and reverse diffusion processes."
>}}

In principle, if we start from pure random noise, we should be able to keep applying the trained model until we obtain a sample that looks as if it were drawn from the training set. That's it -- and yet this simple idea works incredibly well in practice.

*For a more visual understanding, check out **[this article](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models)** -- it provides an interactive, step-by-step introduction that makes diffusion models much easier to grasp.*

## How do diffusion models work?
### Forward diffusion process:
Suppose we have a sample from a real data distribution $\mathbf{x}_0 \sim q(\mathbf{x})$. In the forward diffusion, we gradually corrupt the sample by adding small amounts of Gaussian noise in $T$ steps, producing a sequence of increasingly noisy samples $\mathbf{x}_1, \dots, \mathbf{x}_T$.
The amount of noise added at each step $t$ is controlled by a variance schedule $\\{\beta\_t \in (0, 1)\\}\_{t=1}^T$.
$$
\begin{aligned}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) &= \mathcal{N}(\mathbf{x}\_t; \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}, \beta\_t\mathbf{I}) \\\
q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0) &= \prod^T\_{t=1} q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1})
\end{aligned}
$$

As $t$ increases, the sample $\mathbf{x}_t$ becomes progressively noisier.
Eventually when $T \rightarrow \infty$, $\mathbf{x}_T$ is indistinguishable from random noise.

Mathematically, we can write each step as follows:
$$
\mathbf{x}\_t = \sqrt{1-\beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1}  \quad \quad \text{where }\boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Here, $\mathcal{N}(\cdot,\cdot)$ denotes a normal distribution.
Since the sum of two Gaussian variables with variances $\sigma^2_1$ and $\sigma^2_2$ is also Gaussian with variance $\sigma^2_1+\sigma^2_2.$ Given that $\boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, if $\mathbf{x}\_{t-1}$ has zero mean and unit variance, then so does $\mathbf{x}_{t}$, as $\sqrt{1-\beta\_t}^2 + \sqrt{\beta\_t}^2=1$.
This scaling ensures that the variance remains stable throughout the diffusion process.
This way, if we normalize our original sample $\mathbf{x}\_{0}$ to have zero mean and unit variance, then the sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ will also maintain these properties and $\mathbf{x}_T$ will approximate a standard Gaussian distribution for sufficiently large $T$.

Another nice property of the above process is that we can jump straight from the original $\mathbf{x}_0$ to any noised version of the forward diffusion process $\mathbf{x}_t$ using a clever reparameterization trick.

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

### Reverse Diffusion:
If we can *undo* the above process and sample from $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$, we can reconstruct a true data sample starting from pure Gaussian noise, $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
Unfortunately, the exact posterior $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ is **intractable**, as computing it would require integrating over the entire data distribution.

Instead, we approximate these conditional probabilities with a parameterized model $p_\theta$ (e.g., a neural network). Because $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ is also Gaussian for sufficiently small $\beta_t$​, we can choose $p_\theta$ to be Gaussian. Therefore, we define the reverse process as:

$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) = \mathcal{N}(\mathbf{x}\_{t-1}; \boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t), \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t))
$$

Here, $\boldsymbol{\mu}\_\theta$ and $\boldsymbol{\Sigma}\_\theta$ are outputs of a neural network that predict the mean and variance of the denoised sample at each timestep

{{< figure
  src="images/reverse_process.jpg"
  alt="Diffusion model"
  caption="Figure 4: Reverse diffusion process."
>}}

If we apply the reverse formula for all timesteps $p_\theta(\mathbf{x}\_{0:T})$, we can go from random noise $\mathbf{x}\_T$ to the data distribution:
$$
p\_\theta(\mathbf{x}\_{0:T}) = p(\mathbf{x}\_T) \prod^T\_{t=1} p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)
$$

>*For a deeper dive into the math behind both the forward and reverse diffusion processes, check out these great resources: [**Lil'Log post**](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), [**The AI Summer’s guide**](https://theaisummer.com/diffusion-models/).*

<!--By additionally conditioning the model on timestep $t$, it will learn to predict the Gaussian parameters (meaning the mean $\boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t)$ and the covariance matrix $\boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t)$) for each timestep.-->

#### But how do we train diffusion models?

To train the diffusion model, we optimize the parameters $\theta$ of the reverse process $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ so that it can accurately **denoise** samples at each step.
Rather than directly learning to reconstruct $ \mathbf{x}_0$, it is more efficient to train the model to predict the **noise** $ \boldsymbol{\epsilon}_t$ that was added during the forward process.

The training objective is derived from the variational lower bound (VLB) on the log-likelihood (see the two posts above if you want to know the details). By minimizing this loss, the model learns how to progressively remove noise from any noisy input—allowing it to generate realistic samples starting from pure random noise.

### Diffusion models in environmental system science
Diffusion models are opening new frontiers in environmental and Earth system science. Their ability to generate realistic, high-resolution data makes them powerful tools for simulating and understanding complex natural processes.

For example:

- Climate and Weather Modeling: Diffusion models can generate fine-grained climate patterns or downscale coarse-resolution simulations, improving forecasts and scenario modeling.

- Remote Sensing and Earth Observation: They can reconstruct missing satellite data, denoise cloudy or corrupted imagery, and synthesize realistic environmental maps.

- Geoscientific Data Generation: In areas like hydrology, seismology, or oceanography, diffusion models can generate synthetic yet physically consistent data, supporting model training and uncertainty quantification.

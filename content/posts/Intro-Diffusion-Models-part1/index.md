---
title: "Diffusion Models: Principles and Applications in Earth Sciences - Part 1"
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
ShowWordCount: false
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

There are already plenty of great articles that dive into the details of diffusion models -- and we’ll share some of our favorites along the way. In this series, we'll keep thing simple: we focus on the core principles (this post) and explore how diffusion models are being used in Earth and environmental science, and why those applications are so promising and exciting (see Part 2 - in preparation).
<!--(see [Part 2]({{< relref "../Intro-Diffusion-Models-part2/index.md" >}})).
-->
Let’s get started!

## What are generative models?

> *Generative models are a type of AI systems that learn the underlying structure of existing data and use it to create new content that resembles to the original.*

What does this mean in practice? Suppose we have a dataset containing photos of dogs.
A generative model can study all those images to learn what makes a picture look like a dog - the shapes, colors, textures, and relationships between pixels. Once trained, the model can then generate completely new, realistic images of dogs that did not exist in the original dataset.

Generative models are also ***probabilistic***, meaning they don’t always produce the same output. Instead, they can create many different versions of an image or dataset, all slightly varied but still realistic. This makes them especially useful for creative tasks, predictive simulation, and risk-based scientific modeling.


{{< figure
  src="images/generative_modeling.jpg"
  alt="Generative"
  caption="Figure 1: A generative model learns features from the training dataset and can generate new, high-quality images. Source: Photo from [Tanishq Mathew Abraham](https://x.com/iscienceluvr/status/1592860024657051649)."
>}}

There are different types of generative models, such as [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661) (GANs), [Variational Autoencoders](https://arxiv.org/abs/1312.6114) (VAEs), [flow-based models](https://en.wikipedia.org/wiki/Flow-based_generative_model), and [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model). Each type has its strengths and weaknesses, but diffusion models have recently shown outstanding performance in producing high-quality, realistic results. Their success largely comes from their ability to progressively refine noise, allowing them to capture complex data distributions and produce stable, high-fidelity results without the training instability common in other generative approaches.
We’ll focus on diffusion models in this series.


{{< figure
  src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"
  alt="Generative"
  caption="Figure 2: Computation graphs of prominent generative models. Source: [Lil'Log](https://lilianweng.github.io/)"
>}}


## What are diffusion models?
Diffusion models are inspired by non-equilibrium thermodynamics in physics -- specifically, how particles spread out or "diffuse" over time. The core idea behind them is similar and simple: we gradually corrupt (add noise to) clean data until it becomes completely random, then train a deep learning model to reverse this process and recover the original data.

> *Diffusion models are a class of generative models that learn to reverse a gradual noising process applied to data, enabling them to generate realistic samples from the underlying data distributions by iteratively denoising random noise.*

In other words, diffusion models learn how to "undo" noise. Imagine taking a blurry or noisy satellite image and carefully sharpening it, one small step at a time, until continents and clouds slowly come back into focus. Each step removes a bit of noise, turning random patterns into something meaningful.

In principle, if we start from pure random noise, we should be able to keep applying the trained model until we obtain a sample that looks as if it were drawn from the training set. That's it -- and yet this simple idea works incredibly well in practice.

>*For a more intuitive explanation, check out [this article](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models) -- it provides an interactive, step-by-step introduction that makes diffusion models much easier to grasp.*

Diffusion models come in different forms, depending on how they add and remove noise -- some are probabilistic, while others are deterministic.
One of the most important and widely used approaches is the [Denoising Diffusion Probabilistic Model](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) (DDPM), which has become the basis for many breakthroughs in generative AI.

## How do diffusion models work?
Now, let’s explore how DDPMs actually work.
At their core, DDPMs involve two distinct stochastic processes: a ***forward diffusion pass*** -- where noise is gradually added to data until it becomes purely random, and a ***reverse denoising process*** -- where the model learns to remove that noise step by step to reconstruct the original data.

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

Here, $\mathcal{N}(\cdot,\cdot)$ denotes a normal distribution.
As $t$ increases, the sample $\mathbf{x}_t$ becomes progressively noisier.
Eventually when $T \rightarrow \infty$, $\mathbf{x}_T$ is indistinguishable from random noise.
Mathematically, we can write each step of this process as follows:
$$
\mathbf{x}\_t = \sqrt{1-\beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1} \quad \quad \text{where } \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Since the sum of two Gaussian variables with variances $\sigma^2_1$ and $\sigma^2_2$ is also Gaussian with variance $\sigma^2_1+\sigma^2_2$, and given that $\boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, if $\mathbf{x}\_{t-1}$ has zero mean and unit variance, then so does $\mathbf{x}_{t}$, because $\sqrt{1-\beta\_t}^2 + \sqrt{\beta\_t}^2=1$.
This way, if we normalize our original sample $\mathbf{x}\_{0}$ to have zero mean and unit variance, then the sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ will also maintain these properties and $\mathbf{x}_T$ will approximate a standard Gaussian distribution for sufficiently large $T$.
This scaling ensures that the variance remains stable throughout the diffusion process.

Another nice property of the above process is that we can jump straight from the original sample $\mathbf{x}_0$ to any noised version of the forward diffusion process $\mathbf{x}_t$ using a clever reparameterization trick as below.

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$, then we can write the following:
$$
\begin{aligned}
\mathbf{x}\_t
&= \sqrt{\alpha\_t}\mathbf{x}\_{t-1} + \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon}\_{t-1} \\\
&= {\color{red}\sqrt{\alpha\_t}\( \sqrt{\alpha\_{t-1}}\mathbf{x}\_{t-2} + \sqrt{1 - \alpha\_{t-1}}\boldsymbol{\epsilon}\_{t-2} \)} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
&= {\color{red}\sqrt{\alpha\_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + \sqrt{\alpha\_t (1-\alpha\_{t-1})}\boldsymbol{\epsilon}\_{t-2}} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
&= \sqrt{\alpha\_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + {\color{red}\sqrt{1 - \alpha\_t \alpha\_{t-1}} \bar{\boldsymbol{\epsilon}}\_{t-2} } \\\
&= \dots \\\
&= \sqrt{\alpha\_t \alpha\_{t-1}\dots\alpha\_1} \mathbf{x}\_{0} + \sqrt{1 - \alpha\_t \alpha\_{t-1}\dots\alpha\_1} \bar{\boldsymbol{\epsilon}} \\\
&= \sqrt{\bar{\alpha}\_t}\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\boldsymbol{\epsilon}
\end{aligned}
$$

>***Explanation in words:*** We unroll the update rule step by step, combining the noise terms along the way, so that $\mathbf{x}\_t$ can be written directly in terms of $\mathbf{x}\_0$.
>Note that since $\boldsymbol{\epsilon}\_{t-2}, \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, their weighted sum is also Gaussian with standard deviation $\sqrt{\alpha_t (1-\alpha_{t-1})+(1-\alpha_t)} = \sqrt{1-\alpha_t\alpha_{t-1}}$, and $\bar{\boldsymbol{\epsilon}}\_{t-2} \sim \mathcal{N}(\mathbf{0},\mathbf{I}).$

The forward diffusion process $q$ can therefore be written in closed form as:
$$
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

### Reverse denosing process
The reverse process does the opposite of the forward one: *instead of adding noise, it removes noise, step by step, to gradually recover the original data*.
Once trained, the model can start from pure Gaussian noise and iteratively apply this reverse procedure to generate new samples similar to $\mathbf{x}_0$.

In theory, the reverse diffusion process is defined as $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ -- meaning that given a noisy sample $\mathbf{x}\_t$, we would like to compute the distribution of the previous, slightly less noisy sample $\mathbf{x}\_{t-1}$. However, this distribution is ***intractable*** in practice because it depends on the entire (unknown) data distribution.

{{< figure
  src="images/reverse_process.jpg"
  alt="Diffusion model"
  caption="Figure 4: Reverse diffusion process."
>}}

#### Conditioning trick
Another useful trick in diffusion models is that the reverse transition becomes tractable ***if we condition on the original clean data $\mathbf{x}\_{0}$:***
$$
q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) = \mathcal{N}(\mathbf{x}\_{t-1}; {\tilde{\boldsymbol{\mu}}}(\mathbf{x}\_t, \mathbf{x}\_0), {\tilde{\beta}\_t} \mathbf{I})
$$
where $\tilde{\boldsymbol{\mu}}\_t = {\frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_t \Big)}$ and $\tilde{\beta}\_t = {\frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \beta\_t}$ are closed-form functions derived from the forward process parameters.

>*For a detailed derivation, see [Lil'Log's diffusion models post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#ldm).*

*What does this mean?* It means that during training, since we know $\mathbf{x}\_0$, we can compute the exact noise that was added to get $\mathbf{x}\_t$. This allows us to generate training pairs $(\mathbf{x}\_t, \mathbf{\epsilon})$ where $\mathbf{\epsilon}$ is the exact noise, and train a model to predict this noise.

#### Why do we need deep learning?
However, at generation time, we start from pure Gaussian noise and do not know $\mathbf{x}\_0$.
So we can no longer use the closed-form $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$.
This is where deep learning comes into play. Instead, we learn a neural network $\mathbf{\epsilon}\_\theta(\mathbf{x}\_{t},t)$ that predicts the noise added at each step. From this noise prediction, we compute an estimate of the clean data and use it to approximate the true reverse process:
$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) \approx q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t).
$$

> *At each diffusion step, the neural network predicts the noise inside the current noisy sample, then subtracts it accordingly.*

Since each step in the forward diffusion adds only a small amount of Gaussian noise, the reverse steps can also be modeled as Gaussian transitions:

$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) = \mathcal{N}(\mathbf{x}\_{t-1}; \boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t), \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t))
$$

By applying this reverse transition from $t=T \rightarrow 0$, we gradually transform pure noise $\mathbf{x}\_T$ to a coherent, realistic sample that is similar to $\mathbf{x}\_0$:
$$
p\_\theta(\mathbf{x}\_{0:T}) = p(\mathbf{x}\_T) \prod^T\_{t=1} p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)
$$

Note that although the noise added during the forward diffusion is random, it is not arbitrary -- its structure comes from the underlying data. As a result, by learning to predict and remove this noise accurately, the model implicitly learns the structure of the original image $\mathbf{x}\_0$ and how to reconstruct it from noise.

#### In short:
- **Training:** we know $\mathbf{x}\_0$ → compute true noise → train model to predict it
- **Generation:** we start from pure Gaussian noise → model predicts noise → remove it step by step

### How do diffusion models learn?
Training a diffusion model means teaching it to predict and remove noise at each timestep of the reverse process.
During training, we know both the original clean data $\mathbf{x}\_0$ and its noisy version $\mathbf{x}\_t$ produced by the forward process.
This lets us compute the true noise $\mathbf{\epsilon}$ added at step $t$.

The model's objective is straightforward:
> **Given a noisy sample $\mathbf{x}\_t$ and a timestep $t$, predict the noise $\mathbf{\epsilon}$ that was added.**

The more accurately it can do this, the better it becomes at reversing the diffusion and generating realistic samples.

To train the model, we use a loss function that measures how close the predicted noise is to the true noise:
$$
L = \mathbb{E}\_{\mathbf{x}\_0, \epsilon, t} \left[ \| \epsilon - \epsilon\_\theta(\mathbf{x}\_t, t) \|^2 \right]
$$
where $\mathbf{x}\_0$ is the original clean data, $\epsilon$ is the Gaussian noise added at time step $t$, $\epsilon_\theta(\mathbf{x}\_t, t)$ is the noise predicted by the model, and $\mathbf{x}\_t$ is the noisy data at step $t$.

By minimizing this loss, the model learns to invert each step of the noising process. As training progresses, the model learns to progressively remove noise from any noisy input $\mathbf{x}\_t$, enabling it to generate realistic samples starting from pure random noise.

Behind the scenes, this loss comes from something called the *Variational Lower Bound* (VLB), also called the *evidence lower bound* (ELBO), -- a statistical framework that connects the model’s noise predictions to the overall likelihood of generating real data.
You can think of it as a mathematical way of ensuring the model learns the most likely way to transform random noise into meaningful patterns.

We skip the full derivation here -- instead, we use the simplified noise-prediction loss, which captures the same idea but is much easier and more efficient to train in practice.

>*If you’d like to explore the full mathematical derivation, check out these excellent resources [[1](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), [2](https://theaisummer.com/diffusion-models/), [3](https://arxiv.org/pdf/2208.11970), [4](https://www.arxiv.org/pdf/2510.21890)]. Each provides a detailed explanation of the theory and intuition behind diffusion models.*

**Quick summary:**
- We gradually add noise to data (forward process).
- The model learns to remove the noise (reverse process).
- Training minimizes the gap between true noise and predicted noise.

In Part 2, we will explore how diffusion models are being used in Earth and environmental sciences.


<!--> *Note: This means diffusion models can use different neural network architectures depending on the data type, making them flexible for many scientific fields, including Earth science.*-->

<!--Remember that we only know $\mathbf{x}\_0$ during training -- not at generation time. This is where deep learning comes into play. Instead of computing the true reverse transition, we approximate it using a neural network $p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$. The training objective is to learn parameters $\theta$ such that:-->

<!--Importantly, the model $p\_\theta(\mathbf{x})$ does not directly predict $\mathbf{x}\_{t-1}$, nor even the specific noise added between $\mathbf{x}\_{t-1}$ and $\mathbf{x}\_{t}$. ***Instead, it predicts the entire noise present in $\mathbf{x}\_{t}$***, then removes a fraction of that noise (based on the state of the variance schedule at that timestep) to obtain $\mathbf{x}\_{t-1}$.-->

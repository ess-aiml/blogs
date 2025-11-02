---
title: "Diffusion Models: Principles and Applications in Earth Sciences - Part 1"
description: "A beginner-friendly introduction to diffusion models: from noise to meaningful structure - with insight for Earth science"
summary: "A beginner’s guide to diffusion models"
date: 2025-10-28
tags: ["Diffusion Model", "Generative", "Earth system"]
author: "Phong Le"
series: ["AI-ML"]
showToc: true
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
    caption: Image created by author using AI
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

There are already plenty of great articles that dive into the details of diffusion models -- and we’ll share some of our favorites along the way. In this series, we'll keep things simple: we focus on the core principles (in this post) and explore how diffusion models are being used in Earth and environmental sciences, and why those applications are so promising (Part 2 - in preparation).

Let’s get started!

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## What are generative models?

>*<mark>Generative models are a type of AI system that learn the underlying structure of existing data and use it to create new content that resembles the original.</mark>*

What does this mean in practice? Suppose we have a dataset containing photos of dogs.
A generative model can study all those images to learn what makes a picture look like a dog -- the shapes, colors, textures, and relationships between pixels. Once trained, the model can then generate completely new, realistic images of dogs that did not exist in the original dataset.

Generative models are also ***probabilistic***, i.e., they don’t always produce the same output. Instead, they can create many different versions of an image or dataset, all slightly varied, but still realistic. This makes them especially useful for creative tasks, predictive simulation, and risk-based scientific modeling.


{{< figure
  src="images/generative_modeling.jpg"
  alt="Generative"
  caption="A generative model learns features from the training dataset and can generate new, high-quality images ([Source](https://x.com/iscienceluvr/status/1592860024657051649))."
>}}

There are different types of generative models, such as Generative Adversarial Networks[^Goodfellow:2014] (GANs), Variational Autoencoders[^Kingma2014] (VAEs), flow-based models[^Kingma2018], and diffusion models[^Sohl-Dickstein2015]<sup>,</sup>[^Ho2020]. Each type has its strengths and weaknesses, but diffusion models have recently shown outstanding performance in producing high-quality, realistic results. Their success largely comes from their ability to progressively refine noise, allowing them to capture complex data distributions and produce stable, high-fidelity results without the training instability common in other generative approaches.
We’ll focus on diffusion models in this series.

{{< figure
  src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"
  alt="Generative"
  caption="Computation graphs of prominent generative models. Source: [Lil'Log](https://lilianweng.github.io/)"
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## What are diffusion models?
Diffusion models are inspired by non-equilibrium thermodynamics -- specifically, how particles spread out or "diffuse" over time. ***The core idea behind them is simple:*** we gradually corrupt (add noise to) clean data until it becomes completely random, then train a deep learning model to reverse this process and recover the original data.

> *<mark>Diffusion models are a class of generative models that learn to reverse a gradual noising process applied to data, enabling them to generate realistic samples from the underlying data distributions by iteratively denoising random noise.</mark>*

In other words, diffusion models learn how to "undo" noise. Imagine taking a blurry or noisy satellite image and carefully sharpening it, one small step at a time, until continents and clouds slowly come back into focus. Each step removes a bit of noise, turning random patterns into something meaningful.

In principle, if we start from pure random noise, we should be able to keep applying the trained model until we obtain a sample that looks as if it were drawn from the training set. ***That's it -- and yet this simple idea works incredibly well in practice***.

>*For a more intuitive explanation, check out [this article](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models) -- it provides an interactive, step-by-step introduction that makes diffusion models much easier to grasp.*

Diffusion models come in different forms, depending on how they add and remove noise -- some are probabilistic, while others are deterministic.
One of the most important and widely used approaches is the [Denoising Diffusion Probabilistic Model](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)[^Ho2020] (DDPM), which has become the basis for many breakthroughs in generative AI.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## How do diffusion models work?
Now, let’s explore how DDPMs actually work.
At their core, DDPMs involve two distinct stochastic processes: a ***forward diffusion pass*** -- where noise is gradually added to data until it becomes purely random, and a ***reverse denoising process*** -- where the model learns to remove that noise step by step to reconstruct the original data.

{{< figure
  src="images/diffusion_processes.jpg"
  alt="Diffusion model"
  caption="Forward and reverse diffusion processes."
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
Eventually, when $T \rightarrow \infty$, $\mathbf{x}_T$ is indistinguishable from random noise.
Mathematically, we can write each step of this process as follows:
$$
\mathbf{x}\_t = \sqrt{1-\beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1} \quad \quad \text{where } \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Since the sum of two Gaussian variables with variances $\sigma^2_1$ and $\sigma^2_2$ is also Gaussian with variance $\sigma^2_1+\sigma^2_2$, and given that $\boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, if $\mathbf{x}\_{t-1}$ has zero mean and unit variance, then so does $\mathbf{x}_{t}$, because $\sqrt{1-\beta\_t}^2 + \sqrt{\beta\_t}^2=1$.

In theory, if we normalize the original sample $\mathbf{x}\_{0}$ to have zero mean and unit variance, then the sequence $\mathbf{x}_1, \dots, \mathbf{x}_T$ will also maintain these properties, and $\mathbf{x}_T$ will approximate a standard Gaussian distribution for sufficiently large $T$. This scaling ensures that the variance remains stable throughout the diffusion process.

In practice, real data is not necessarily Gaussian, and diffusion models do not require Gaussian statistics on the input. Instead, inputs are typically scaled to a bounded range (e.g., [0,1] or [-1,1]). This range must be known and consistent because the noise schedule is defined based on a specific data scale. By the [central limit theorem (CTL)](https://en.wikipedia.org/wiki/Central_limit_theorem), repeatedly adding Gaussian noise ensures that the distribution of $\mathbf{x}_t$ approaches a Gaussian regardless of the initial data distribution.

><mark>Stability in practical diffusion models comes from the variance (noise) schedule -- not from forcing the data to be Gaussian.

<!--The forward diffusion process then gradually drives the distribution of $\mathbf{x}\_{t}$ toward a standard Gaussian as $t$ increases, independent of the original data distribution. Thus, stability in practical diffusion models is achieved through the noise schedule and learned reverse process, rather than strict dataset standardization.-->

Another nice property of the above process is that we can jump straight from the original sample $\mathbf{x}_0$ to any noised version of the forward diffusion process $\mathbf{x}_t$ using a clever reparameterization trick as follow.

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$, then we can write the following:
$$
\begin{aligned}
\mathbf{x}\_t
&= \sqrt{\alpha\_t}\mathbf{x}\_{t-1} + \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon}\_{t-1} \\\
&= \sqrt{\alpha\_t} {\color{red}\(\underbrace{\sqrt{\alpha\_{t-1}}\mathbf{x}\_{t-2} + \sqrt{1 - \alpha\_{t-1}}\boldsymbol{\epsilon}\_{t-2}}_{\color{black}\mathbf{x}\_{t-1}} \)} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
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

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Reverse denoising process
The reverse process does the opposite of the forward one: ***instead of adding noise, it removes noise, step by step, to gradually recover the original data***.
Once trained, the model can start from pure Gaussian noise and iteratively apply this reverse procedure to generate new samples similar to $\mathbf{x}_0$.

In theory, the reverse diffusion process is defined as $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ -- meaning that given a noisy sample $\mathbf{x}\_t$, we would like to compute the distribution of the previous, slightly less noisy sample $\mathbf{x}\_{t-1}$. However, this distribution is ***intractable*** in practice because it depends on the entire (unknown) data distribution.

{{< figure
  src="images/reverse_process.jpg"
  alt="Diffusion model"
  caption="Reverse diffusion process."
>}}

#### Conditioning trick
Another useful trick in diffusion models is that the reverse transition becomes tractable if we condition on the original clean data $\mathbf{x}\_{0}$:
$$
q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) = \mathcal{N}(\mathbf{x}\_{t-1}; {\tilde{\boldsymbol{\mu}}}(\mathbf{x}\_t, \mathbf{x}\_0), {\tilde{\beta}\_t} \mathbf{I})
$$
where $\tilde{\boldsymbol{\mu}}\_t = {\frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_t \Big)}$ and $\tilde{\beta}\_t = {\frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \beta\_t}$, which are closed-form functions derived from the forward process parameters.

>*For a detailed derivation of this trick, see [Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models) diffusion models post[^lillog_diff].*

*What does this mean?* It means that during training, since we know $\mathbf{x}\_0$, we can compute the exact noise that was added to get $\mathbf{x}\_t$. This allows us to create training pairs $(\mathbf{x}\_t, \mathbf{\epsilon})$ where $\mathbf{\epsilon}$ is the exact noise, and train a model to predict this noise.

#### Why do we need deep learning?
However, at generation time, we start from pure Gaussian noise and do not know $\mathbf{x}\_0$.
So we can no longer use the closed-form $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$.

This is where deep learning comes into play.
We instead train a neural network $\mathbf{\epsilon}\_\theta(\mathbf{x}\_{t},t)$ to predict the noise added at each step. Once we have this noise estimate, we can recover an estimate of the clean signal and approximate the true reverse process:
$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) \approx q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t).
$$

> *<mark>At each diffusion step, the neural network predicts the noise inside the current noisy sample and then subtracts it accordingly.</mark>*

Since each step in the forward diffusion adds only a small amount of Gaussian noise, the reverse steps can also be modeled as Gaussian transitions:

$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) = \mathcal{N}(\mathbf{x}\_{t-1}; \boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t), \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t))
$$

By applying this reverse transition from $t=T \rightarrow 0$, we gradually transform pure noise $\mathbf{x}\_T$ to a coherent, realistic sample that is similar to $\mathbf{x}\_0$:
$$
p\_\theta(\mathbf{x}\_{0:T}) = p(\mathbf{x}\_T) \prod^T\_{t=1} p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)
$$

Note that although the noise added during the forward diffusion is random, it is not arbitrary -- its structure comes from the underlying data. As a result, by learning to predict and remove this noise accurately, the model implicitly learns the structure of the original image $\mathbf{x}\_0$ and how to reconstruct it from noise.

**In short:**
>- **Training:** we know $\mathbf{x}\_0$ → compute true noise → train model to predict it
>- **Generation:** we start from pure Gaussian noise → model predicts noise → remove it step by step

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Train diffusion models
The goal of training a diffusion model is to make it assign ***high probability to real data***. Formally, we want to maximize the likelihood of samples from the true data distribution:
$$
\max\_{\theta} \mathbb{E}\_{\mathbf{x}\_0 \sim q(\mathbf{x}\_0)} \left[ \log p\_{\theta}(\mathbf{x}\_0) \right]
$$
Here $q(\mathbf{x}\_0)$ is the real data distribution, and $p\_{\theta}(\mathbf{x}\_0)$ is the distribution modeled by the neural network.
However, the likelihood $p\_{\theta}(\mathbf{x}\_0)$ is intractable because the model generates data through a chain of latent noisy variables:
$$
p_{\theta}(x_0)
= \int p\_\theta(\mathbf{x}\_{0:T}) dx_{1:T}
$$

To solve this, diffusion models use a classical idea from variational inference -- the Evidence Lower Bound (ELBO).

><mark>We can’t compute the true likelihood, but we can compute a lower bound on it and train the model by maximizing that bound.</mark>

ELBO is a computable lower bound on the true log-likelihood of data. We maximize it because doing so also maximizes the likelihood of real data — but in a way we can actually calculate.

$$
\begin{aligned}
\underbrace{\log p\_\theta(\mathbf{x}\_0)}\_{\text{Evidence}}
&= \log \int p\_\theta(\mathbf{x}\_{0:T}) dx_{1:T} = \log \int {\color{red} q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})} \frac{p\_\theta(\mathbf{x}\_{0:T})}{\color{red}{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})}} dx_{1:T} \\\
&= \log \mathbb{E}\_{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0)} \Bigg[\frac{p\_\theta(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})}\Bigg] \quad \quad \color{green}\small{\text{By definition: } \mathbb{E}\_{p(x)}[f(x)] = \int p(x)f(x)dx} \\\
&\ge \underbrace{\mathbb{E}\_{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0)} \Bigg[ \log \frac{p\_\theta(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})} \Bigg]}\_{\text{Evidence Lower Bound (ELBO)}} \quad \quad \color{green}\small{\text{Apply Jensen's inequality (log is concave)}} \\\
&\rule{0pt}{1.5em} \color{blue}\text{...  see references for detailed derivation, at the end we obtain:} \\\
&\ge \underbrace{\mathbb{E}\_{q(\mathbf{x}\_{1} \vert \mathbf{x}\_0)} \Big[ \log p_{\theta}(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0}) - D\_\text{KL}(q(\mathbf{x}\_{T}\vert\mathbf{x}\_0) \|| p\_\theta(\mathbf{x}\_{T}) ) - \sum\_{t=2}^T D\_\text{KL}(q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_{t-1} \vert\mathbf{x}\_t)) \Big]}\_{\text{Variational Lower Bound ($L\_{VLB}$)}}
\end{aligned}
$$

where $D\_\text{KL}(p||q)$ is the Kullback–Leibler (KL) divergence. It measures the similarity between two distributions. KL divergence is always positive and can be non-symmetric under the interchange of $p$ and $q$.

To train the model, we instead minimize the negative log-likelihood bound:
$$
-\log p\_\theta(\mathbf{x}\_0) \le \mathbb{E}\_{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0)} \Big[\underbrace{- \log p\_\theta(\mathbf{x}\_0 \vert \mathbf{x}\_1)}\_{L\_0} + \sum\_{t=2}^T \underbrace{D\_\text{KL}(q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_{t-1} \vert\mathbf{x}\_t))}\_{L\_{t-1}} + \underbrace{D\_\text{KL}(q(\mathbf{x}\_T \vert \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_T))}\_{L\_T} \Big]
$$

Every KL divergence term in $L_{LVB}$ (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in closed form. $L_T$ is constant and can be ignored during training.

In DDPMs, this ultimately reduces to a simple and intuitive loss:
$$
\mathcal{L}(\theta) = \mathbb{E}\_{\mathbf{x}\_0, \epsilon, t} \left[ || \epsilon - \epsilon\_\theta(\mathbf{x}\_t, t) ||^2 \right]
$$

By minimizing this loss, the model learns to invert each step of the noising process. As training progresses, it becomes increasingly effective at removing noise from any noisy input $\mathbf{x}\_T$, enabling it to generate realistic samples starting from pure random noise.

>*If you’d like to explore the complete mathematical derivation, check out these excellent resources[^lillog_diff]<sup>,</sup>[^theaisummer]<sup>,</sup>[^Lai2025]<sup>,</sup>[^Luo2022]<sup>,</sup>[^Ozdemir]. Each provides a detailed explanation of the theory and intuition behind diffusion models.*

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

**Quick summary:**
- We gradually add noise to data (forward process).
- The model learns to remove the noise (reverse process).
- Training is to maximize log-likelihood of real data.
- Exact likelihood is intractable, so we optimize a lower bound (ELBO) instead.
- Once trained, the model can start from random noise and iteratively denoise to generate realistic samples.

In Part 2, we'll dive into how diffusion models are applied in Earth and environmental sciences -- stay tuned!

## References
[^Goodfellow:2014]: Goodfellow, I. et al., 2014. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). *Advances in Neural Information Processing Systems (NeurIPS)*, 27, pp.2672–2680.
[^Kingma2014]: Kingma, D.P. & Welling, M., 2014. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). *Proceedings of the International Conference on Learning Representations (ICLR)* 2014.
[^Kingma2018]: Kingma, D.P. & Dhariwal, P., 2018. [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039). *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, Montréal, Canada.
[^Sohl-Dickstein2015]: Sohl-Dickstein, J. et al., 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585). *Proceedings of the 32$^{nd}$ International Conference on Machine Learning (ICML)*, PMLR, 37, pp.2256–2265.
[^Ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 6840-6851.
[^lillog_diff]: https://lilianweng.github.io/posts/2021-07-11-diffusion-models
[^theaisummer]: https://theaisummer.com/diffusion-models
[^Lai2025]: Lai, C.-H. et al., 2025. [The Principles of Diffusion Models](https://www.arxiv.org/pdf/2510.21890).
[^Luo2022]: Luo, C., 2022. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970).
[^Ozdemir]: Özdemir H., [Diffusion Models Explained with Math From Scratch](https://www.youtube.com/watch?v=fbJac4qQy04).

---
title: "Diffusion Models: Principles and Applications in Earth Sciences - Part 2"
summary: "Diffusion models for environmental science"
date: 2026-10-31
tags: ["Diffusion Model", "Environment", "Earth system"]
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

There are already plenty of great articles that dive into the details of diffusion models -- and we’ll share some of our favorites along the way. In this post, we'll keep thing simple: we focus on the core principles (part 1) and explore how diffusion models are being used in environmental science, and why those applications are so promising and exciting (this post).

Let’s get started!

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

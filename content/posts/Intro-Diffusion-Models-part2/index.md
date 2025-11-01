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
    # image: "images/Earth_diffusion.jpg" # image path/url
    image: "https://cdn.satnow.com/community/AI_imagery_cover_638882532883884551.png"
    alt: "<alt text>" # alt text
    caption: "Source: satnow.com" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-aiml/blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

In [Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), we explored the principles of diffusion models -- how they take random noise and gradually transform it into meaningful data. In this post, we'll look at how diffusion models are being used in Earth science, and why those applications are so promising.

>*Diffusion models are reshaping how we explore and understand Earth’s systems.*

### Complexity in Earth system
Earth’s components -- atmosphere, oceans, land, and cryosphere -- interact in nonlinear and chaotic ways. These interactions produce both familiar seasonal patterns and extreme events like floods, droughts, atmospheric rivers, and heatwaves. To manage our planet in a changing climate, we need models that can capture uncertainty, extremes, and multi-scale variability.

<mark>The five systems of Earth</mark> (geosphere, biosphere, cryosphere, hydrosphere, and atmosphere) interact to produce the environments we are familiar with.

<cite>[^1]</cite>

Understanding and modeling these dynamics are essential. They help us:
- Prepare for natural hazards
- Optimize renewable energy systems
- Manage water and agricultural resources
- Design climate-resilient infrastructure
- and so on

>**Question**: Can diffusion models help us to solve these problems?
>*Probably, yes*

### Why diffusion models for environmental system science?
Diffusion models bring several unique strengths to environmental science:
- Generative by design: they don’t just predict one outcome; they create many plausible scenarios, perfect for uncertainty and risk-based research.
- Excellent at capturing complex distributions — environmental data is noisy, skewed, multi-scale, and often chaotic. Diffusion models handle this complexity naturally.

>*Diffusion models don’t just predict -- they explore possibilities.*


{{< figure
  src="images/Guilloteau2024.gif"
  alt="Diffusion model"
  caption="Diffusion-based Ensemble Rainfall estimation from Satellite (DifERS). Source [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)"
>}}

An example comes from [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662), who developed a generative diffusion framework called DifERS for producing ensembles of precipitation maps from multisensor satellite data. Their method combines physical insight with statistical learning to reconstruct detailed rainfall patterns from coarse satellite inputs. Two novelties of their method thus are: 1) the handling of the uncertainty through the generation of ensembles of equiprobable realizations and 2) the use of coincident measurements from different instruments and different platforms.

Because diffusion models explicitly model the distribution of states rather than just the mean response, they are particularly well-suited for capturing uncertainty, extremes, and multi-scale variability—features that are notoriously difficult for traditional deep learning architectures. For instance, recent studies [[Bassetti et al, (2024)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194), [Hess et al, (2025)](https://www.nature.com/articles/s42256-025-00980-5)] have shown that diffusion-based emulators can reconstruct fine-scale rainfall structures from coarse reanalysis data while preserving the physical coherence of storm systems, something most conventional downscaling models tend to blur.

**To be continue...**

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.

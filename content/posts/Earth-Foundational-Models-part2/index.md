---
title: "Earth Foundational Models - An Introduction (Part 2) -- Test"
date: 2025-10-06
tags: ["Foundational Model"]
summary: "A beginner’s guide to the powerful models shaping the future of Earth science"
series: ["PaperMod"]
# weight: 1
# aliases: ["/papermod-installation"]
tags: ["Foundational Model"]
author: ["AI/ML WG"]
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
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
cover:
  image: images/DOFA1.png
  alt: "<alt text>" # alt text
  caption: "The Dynamic One-For-All (DOFA) model is a multimodal foundation model tailored for remote sensing and Earth observation tasks (Source: esri.com)"
  # relative: false # when using page bundles set this to true
  # hidden: true # only hide on current single page
  hiddenInList: true
social:
  fediverse_creator: "@Phong Le - ORNL"
editPost:
    URL: "https://github.com/ess-ai-ml/content/posts/Earth-Foundational-Models-part2/index.md"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

> Foundation models are powerful artificial intelligence (AI) models that are trained on a massive amount of data and can be adapted to a wide range of tasks.

There are 5 key characteristics of Foundation Models:

1. Pretrained (using large data and massive compute so that it is ready to be used without any additional training)
2. Generalized — one model for many tasks (unlike traditional AI which was specific for a task such as image recognition)
3. Adaptable (through prompting — the input to the model using say text)
4. Large (in terms of model size and data size e.g. GPT-3 has 175B parameters and was trained on about 500,000 million words, equivalent of over 10 lifetimes of humans reading nonstop!)
5. Self-supervised (see footnote 1) — no specific labels are provided and the model has to learn from the patterns in the data which is provided — see the cake illustration below.

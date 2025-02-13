---
layout: post
title: Understanding, Estimating, and Controlling Log-Density in Diffusion Models
date: 2025-02-11 10:25:00
description: Understanding, Estimating, and Controlling Log-Density in Diffusion Models
bibliography: blogs.bib
tags: diffusion
categories: sample-posts
---

# Understanding, Estimating, and Controlling Log-Density in Diffusion Models

## What is Log-Density?

Log-density, or the log-likelihood of a sample under a generative model, often serves as a proxy for how "typical" or "in-distribution" a sample is. Since diffusion models are likelihood-based models<citation>, they aim to assign high likelihood to training data and, by extension, low likelihood to out-of-distribution (OOD) data. Intuitively, one might think that log-density is a reliable measure of whether a sample lies in or out of the data distribution.

However, prior research <citation> has shown that generative models can sometimes assign higher likelihoods to OOD data than to in-distribution data. In <d-cite key="karczewski2025diffusion"></d-cite>, we show that diffusion models are no different. In fact, we push this analysis further by exploring the highest-density regions of diffusion models.

Using a theoretical **mode-tracking ODE**, we investigate the regions of the data space where the model assigns the highest likelihood. Surprisingly, these regions are often occupied by cartoon-like drawings or blurry images—patterns that are absent from the training data. Additionally, we observe a strong correlation between negative log-density and PNG image size, revealing that negative log-likelihood for image data is essentially a measure of **information content** or **detail**, rather than "in-distribution-ness".

---

## How to Measure Log-Density?

To measure log-density in diffusion models, it’s important to understand the modes of sampling available. Broadly, sampling in diffusion models can be categorized into two dimensions:

1. **Deterministic vs Stochastic Sampling**:
   - Deterministic sampling uses the **probability flow ODE**, following a smooth trajectory.
   - Stochastic sampling leverages the **reverse SDE**, introducing randomness at each step.

2. **Original Dynamics vs Modified Dynamics**:
   - Sampling can follow the original dynamics dictated by PF-ODE or Rev-SDE.
   - Alternatively, one can modify the dynamics to control specific properties, such as the log-density trajectory.

We propose a **quadrant framework** to organize these methods, as shown below:

| Sampling Mode     | Original Dynamics       | Any Dynamics       |
|--------------------|-------------------------|--------------------------|
| **Deterministic** | Known from prior work <d-cite key="chen2018neural"></d-cite>  | Introduced in <d-cite key="karczewski2025diffusion"></d-cite>     |
| **Stochastic**    | Introduced in <d-cite key="karczewski2025diffusion"></d-cite>    | Introduced in <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>     |

Previously, log-density was only measurable for deterministic sampling with original dynamics. In <d-cite key="karczewski2025diffusion"></d-cite>, we extend this to deterministic sampling under modified dynamics and stochastic sampling under original dynamics.<d-footnote> Interestingly, we show in <d-cite key="karczewski2025diffusion"></d-cite> that once the true score function is replaced with the approximate one, the log-density estimate becomes biased. We derive the exact formula for this bias and show that it goes to zero when the score function estimation error does.</d-footnote>
In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we further generalize this to stochastic sampling with modified dynamics, deriving the evolution of log-density using the general **Itô's Lemma** and the **Fokker-Planck equation**.
One can see that since stochastic trajectories are a strict generalization of determinic ones (vanishing diffusion term), the method for log-density estimation for any stochastic trajectory is a strict generalization of all the other ones.

---

## How to Control Log-Density?

### Understanding the Connection Between Log-Density and Image Detail

Our findings show that log-density in image models correlates strongly with the amount of detail in the image. Higher likelihood samples tend to exhibit fewer details, often appearing smooth or even cartoon-like. Conversely, lower likelihood samples are richer in detail, capturing complex textures and structures.

An interesting observation <citation> is that simply rescaling the latent code (e.g., scaling the noise at the start of the sampling process) changes the amount of detail in the generated image. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we provide a theoretical explanation for this phenomenon using a concept we call Score Alignment, which directly ties the scaling of the latent code to changes in log-density.

#### Score Alignment

Score alignment measures the angle between the score function at \(t = T\) (the noise distribution) pushed forward via the flow to \(t = 0\), and the score function at \(t = 0\) (the data distribution). If the angle is always acute, scaling the latent code at \(t = T\) changes \(\log p_0(x_0)\) in a monotonic way, explaining the relationship between scaling and image detail.<d-footnote> If the angle is always obtuse, the reverse relationship between logpT and logp0 holds. </d-footnote> Remarkably, we show that this alignment can be measured without explicitly knowing the score function.

### Density Guidance: A Principled Approach to Controlling Log-Density

While latent code scaling provides a heuristic way to control image detail, it lacks precision. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we introduce **Density Guidance**, a principled modification of the generative ODE that allows precise control over the evolution of log-density during sampling.

The key idea is to design dynamics such that:

\[
\frac{\mathrm{d} \log p_t(x_t)}{\mathrm{d} t} = b_t(x_t),
\]

where \(b_t\) is a predefined function specifying how the log-density should evolve. Our method minimizes deviations from the original sampling trajectory while ensuring the desired log-density changes. Empirically, this produces results similar to latent code scaling but with far greater flexibility and control.

#### Choosing Dynamics

While density guidance theoretically allows arbitrary changes to log-density, practical constraints must be considered. Log-density changes that are too large or too small can lead to samples falling outside the typical regions of the data distribution. To address this, we leverage an observation that the following term:

\[
h(x) = \frac{\sigma_t^2 (\Delta \log p_t(x_t)) + \|\nabla \log p_t(x_t)\|^2}{\sqrt{2D}}
\]

is approximately \(N(0, 1)\) for high-dimensional data. This helps determine the "typical" range of log-density changes.

### Stochastic Sampling with Density Guidance

So far, we’ve discussed controlling log-density in deterministic settings. However, stochastic sampling introduces additional challenges and opportunities. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we extend density guidance to stochastic dynamics, showing that log-density can evolve smoothly under predefined trajectories, even when noise is injected.

This is particularly useful for balancing detail and variability in generated samples. For example:
- Adding noise early in the sampling process introduces variation in high-level features like shapes.
- Adding noise later affects only low-level details like texture.

Our method ensures that the log-density evolution remains consistent with the deterministic case, allowing precise control while injecting controlled randomness.

---

## Conclusion

Log-density is a crucial concept in understanding and controlling diffusion models. It measures the level of detail in generated images rather than merely determining in-distribution likelihood. In <d-cite key="karczewski2025diffusion"></d-cite>, we explore the curious behavior of high-density regions in diffusion models, revealing unexpected patterns and proposing new ways to measure log-density across sampling methods. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we go further, introducing **Density Guidance** and demonstrating how to precisely control log-density in both deterministic and stochastic settings.

These findings not only advance our theoretical understanding of diffusion models but also open up practical avenues for generating images with fine-grained control over detail and variability.

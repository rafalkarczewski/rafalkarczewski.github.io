---
layout: distill
title: Understanding Likelihood in Diffusion Models
date: 2025-02-11 10:25:00
description: A summary of our two recent articles on estimating and controling likelihood in diffusion models.
bibliography: blogs.bib
related_publications: true
hidden: true
pretty_table: true
---

## What is Log-Density?

Log-density, or the log-likelihood of a sample under a generative model, often serves as a proxy for how "typical" or "in-distribution" a sample is. Since diffusion models are likelihood-based models <d-cite key="song2021maximum,kingma2024understanding"></d-cite>, they aim to assign high likelihood to training data and, by extension, low likelihood to out-of-distribution (OOD) data. Intuitively, one might think that log-density is a reliable measure of whether a sample lies in or out of the data distribution.

However, prior research <d-cite key="choi2018waic,nalisnick2018deep,nalisnick2019detecting,ben2024d"></d-cite> has shown that generative models can sometimes assign higher likelihoods to OOD data than to in-distribution data. In <d-cite key="karczewski2025diffusion"></d-cite>, we show that diffusion models are no different. In fact, we push this analysis further by exploring the highest-density regions of diffusion models.

Using a theoretical **mode-tracking ODE**, we investigate the regions of the data space where the model assigns the highest likelihood. Surprisingly, these regions are occupied by cartoon-like drawings or blurry images—patterns that are absent from the training data. Additionally, we observe a strong correlation between negative log-density and PNG image size, revealing that negative log-likelihood for image data is essentially a measure of **information content** or **detail**, rather than "in-distribution-ness".

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/cats_logp.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Likelihood measures the amount of detail in an image. </figcaption>
</div>

### Why Does This Happen?

The observation that blurry images or cartoon-like drawings have the highest density in diffusion models may seem counterintuitive. Does this imply that the model considers these images to be the "most likely"? To understand this phenomenon, it is crucial to distinguish between **probability density** and **probability**.

#### Probability Density vs. Probability

A helpful analogy is the standard normal Gaussian distribution in high dimensions. Its density is proportional to $$\exp(-\|\mathbf{x}\|^2 / 2) $$ at any $$\mathbf{x} \in \mathbb{R}^D$$, which is the highest at the origin (zero vector).
However, the origin is actually far from the "typical region" of the distribution. Most samples from a high-dimensional Gaussian are not concentrated near the origin but instead fall in a region at a certain distance from it. Why is that?

The probability of being in a region $$ A $$ is given by the integral of the density over that region:

$$
P(A) = \int_{A} p(\mathbf{x}) d\mathbf{x}.
$$

If the density is constant within $$ A $$, then the probability equals the product of the density and the volume of $$A $$: 

$$
P(A) = c \cdot \text{Vol}(A).
$$

It is crucial to consider both the density and the volume!

#### Gaussian Example: High Density but Low Probability at the Origin

If we compute the probability of a thin spherical shell at radius $$ r $$ and thickness $$ dr $$, the volume of this shell is proportional to $$ r^{D-1}dr $$, and the probability is given by:

$$
P(\text{shell at } r) \propto r^{D-1} \exp(-r^2 / 2)dr.
$$

The key insight is that this probability is maximized not at $$ r = 0 $$ (the origin, where density is highest), but at $$ r = \sqrt{D-1} $$. The typical region is the sweet spot, where neither the volume nor the density is too low.

#### Diffusion Models: High-Density Blurry Images vs. High-Volume Detailed Images

A similar principle applies to diffusion models. Although blurry or cartoon-like images occupy regions of high density, the "volume" of such images—i.e., the diversity of possible variations—is much smaller compared to the volume of regions corresponding to detailed, textured images. As a result, diffusion models assign lower log-densities to more detailed images.

<!-- ---

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/gaussian_vs_diffusion.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Left: In a high-dimensional Gaussian, the highest density occurs at the origin, but the probability mass concentrates in a spherical shell. Right: Similarly, in diffusion models, blurry or cartoon-like images correspond to high-density regions, while detailed images lie in high-probability regions due to their larger volume. </figcaption>
</div>

--- -->


## How to Measure Log-Density?

To measure log-density in diffusion models, it’s important to understand different modes of sampling. Broadly, sampling in diffusion models can be categorized into two dimensions:

1. **Deterministic vs Stochastic Sampling**:
   - Deterministic sampling uses smooth trajectories given by ODEs.
   - Stochastic sampling uses noisy trajecotories given by SDEs.

2. **Original Dynamics vs Modified Dynamics**:
   - Sampling can follow the original dynamics dictated by PF-ODE or Rev-SDE.
   - Alternatively, one can modify the dynamics to control specific properties, such as the log-density trajectory.

We summarize these in the table below:

| Sampling Mode     | Original Dynamics       | Any Dynamics       |
|--------------------|-------------------------|--------------------------|
| **Deterministic** | Prior work <d-cite key="chen2018neural"></d-cite>  | Ours <d-cite key="karczewski2025diffusion"></d-cite>     |
| **Stochastic**    | Ours <d-cite key="karczewski2025diffusion"></d-cite>    | Ours <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>     |

Previously, log-density was only measurable for deterministic sampling with original dynamics. In <d-cite key="karczewski2025diffusion"></d-cite>, we extend this to deterministic sampling under modified dynamics and stochastic sampling under original dynamics.<d-footnote> Interestingly, we show in <d-cite key="karczewski2025diffusion"></d-cite> that once the true score function is replaced with the approximate one, the log-density estimate becomes biased. We derive the exact formula for this bias and show that it goes to zero when the score function estimation error does.</d-footnote>
In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we further generalize this to stochastic sampling with modified dynamics, deriving the evolution of log-density using the general **Itô's Lemma** and the **Fokker-Planck equation**.
One can see that since stochastic trajectories are a strict generalization of determinic ones (vanishing diffusion term), the method for log-density estimation for any stochastic trajectory is a strict generalization of all the other ones.

## How to Control Log-Density?

### Understanding the Connection Between Log-Density and Image Detail

Our findings show that log-density in image models correlates strongly with the amount of detail in the image. Higher likelihood samples tend to exhibit fewer details, often appearing smooth or even cartoon-like. Conversely, lower likelihood samples are richer in detail, capturing complex textures and structures.

An interesting observation <d-cite key="song2021scorebased"></d-cite> is that simply rescaling the latent code (e.g., scaling the noise at the start of the sampling process) changes the amount of detail in the generated image. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we provide a theoretical explanation for this phenomenon using a concept we call Score Alignment, which directly ties the scaling of the latent code to changes in log-density.

#### Score Alignment

Score alignment measures the angle between the score function at $$t = T$$ (the noise distribution) pushed forward via the flow to $$t = 0$$, and the score function at $$t = 0$$ (the data distribution). If the angle is always acute, scaling the latent code at $$t = T$$ changes $$\log p_0(\mathbf{x}_0)$$ in a monotonic way, explaining the relationship between scaling and image detail.<d-footnote> If the angle is always obtuse, scaling has a reverse effect. </d-footnote> Remarkably, we show that this alignment can be measured without explicitly knowing the score function, see <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> for the proof and JAX code.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/sa_vis.png">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Score Alignment is a condition that guarantees monotonic impact of scaling the latent code on the log-density of the decoded sample. It is tractable to verify in practice, even without knowing the score function. Empirically we verify that it almost always holds for diffusion models on image data. </figcaption>
</div>

### Density Guidance: A Principled Approach to Controlling Log-Density

While latent code scaling provides a way to control image detail, it lacks precision. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we introduce **Density Guidance**, a principled modification of the generative ODE that allows precise control over the evolution of log-density during sampling.

The key idea is to design dynamics of $$\mathbf{x}_t$$ such that:

\begin{equation}\label{eq:logp-b}
\frac{d \log p_t(\mathbf{x}_t)}{d t} = b_t(\mathbf{x}_t),
\end{equation}

where $$b_t$$ is a predefined function specifying how the log-density should evolve. We show that, for a flow model $$d \mathbf{x}_t=\mathbf{u}_t(\mathbf{x}_t)dt $$, which defines marginal distributions $$ p_t $$, we can modify the vector field to achieve \eqref{eq:logp-b} as long as we know the score $$ \nabla \log p_t(\mathbf{x})$$:

$$
\tilde{\mathbf{u}}_t(\mathbf{x})=\mathbf{u}_t(\mathbf{x}) + \underbrace{\frac{\operatorname{div}\mathbf{u}_t(\mathbf{x}) + b_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|^2}\nabla \log p_t(\mathbf{x})}_{\text{log-density correction}}.
$$
 
Our method minimizes deviations from the original sampling trajectory while ensuring the desired log-density changes. Empirically, this produces results similar to latent code scaling but with far greater flexibility and control.

#### Choosing Dynamics

While density guidance theoretically allows arbitrary changes to log-density, practical constraints must be considered. Log-density changes that are too large or too small can lead to samples falling outside the typical regions of the data distribution. To address this, we leverage an observation that the following term:

$$
h_t(\mathbf{x}) = \frac{\sigma_t^2 \left(\Delta \log p_t(\mathbf{x}) + \|\nabla \log p_t(\mathbf{x})\|^2\right)}{\sqrt{2D}}
$$

is approximately $$\mathcal{N}(0, 1)$$ for $$\mathbf{x} \sim p_t$$, where the data dimension $$D$$ is high. This helps determine the "typical" range of log-density changes.

### Stochastic Sampling with Density Guidance

So far, we’ve discussed controlling log-density in deterministic settings. However, stochastic sampling introduces additional challenges and opportunities. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we extend density guidance to stochastic dynamics, showing that log-density can evolve smoothly under predefined trajectories, even when noise is injected.

This is particularly useful for balancing detail and variability in generated samples. For example:
- Adding noise early in the sampling process introduces variation in high-level features like shapes.
- Adding noise later affects only low-level details like texture.

Our method ensures that the log-density evolution remains consistent with the deterministic case, allowing precise control while injecting controlled randomness.

## Conclusion

Log-density is a crucial concept in understanding and controlling diffusion models. It measures the level of detail in generated images rather than merely determining in-distribution likelihood. In <d-cite key="karczewski2025diffusion"></d-cite>, we explore the curious behavior of high-density regions in diffusion models, revealing unexpected patterns and proposing new ways to measure log-density across sampling methods. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we go further, introducing **Density Guidance** and demonstrating how to precisely control log-density in both deterministic and stochastic settings.

These findings not only advance our theoretical understanding of diffusion models but also open up practical avenues for generating images with fine-grained control over detail and variability.

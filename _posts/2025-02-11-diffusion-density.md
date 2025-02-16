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

## Diffusion models recap

The idea of diffusion models <d-cite key="sohl2015deep,ho2020denoising,song2021scorebased"></d-cite> is to gradually transform the data distribution $$p_0$$ into pure noise $$p_T$$ (e.g. $$\mathcal{N}(0, I)$$). This is achieved via the forward noising kernel $$p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 I)$$ with $$\alpha, \sigma$$ chosen so that all the information is lost at $$t=T$$, i.e. $$p_T(\mathbf{x}_T \mid \mathbf{x}_0) \approx p_T(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \sigma_T^2 I)$$.
Hence, as $$t$$ increases, $$\mathbf{x}_t$$​ becomes more `noisy', and at $$t=T$$ we reach a tractable distribution $$p_T$$.
This process can equivalently be written as a Stochastic Differential Equation (SDE):

\begin{equation}
d\mathbf{x}_t = f(t)\mathbf{x}_t dt + g(t)d W_t, 
\end{equation}

where $$f, g$$ are scalar functions and $$W$$ is the Wiener process.
Remarkably, this process is reversible!
The Reverse SDE <d-cite key="anderson1982reverse"></d-cite> is given by

\begin{equation}\label{eq:rev-sde}
d\mathbf{x}_t = (f(t)\mathbf{x}_t - g^2(t)\nabla \log p_t(\mathbf{x}_t)) dt + g(t)d \overline{W}_t,
\end{equation}

where $$\overline{W}$$ is the Wiener process going backwards in time and $$\nabla \log p_t(\mathbf{x}_t)$$ is the *score function*, which can be accurately approximated with a neural network <d-cite key="hyvarinen2005estimation,vincent2011connection,song2020sliced"></d-cite>.
Since $$p_T$$ is a tractable distribution, we can easily sample $$\mathbf{x}_T \sim p_T$$ and solve \eqref{eq:rev-sde} to generate a sample $$\mathbf{x}_0 \sim p_0$$.

Rather surprisingly, it turns out that there exists an equivalent *deterministic* process <d-cite key="song2021scorebased,song2020denoising"></d-cite> given by an Ordinary Differential Equation (ODE):

\begin{equation}\label{eq:pf-ode}
d\mathbf{x}_t = (f(t)\mathbf{x}_t - \frac{1}{2}g^2(t)\nabla \log p_t(\mathbf{x}_t)) dt,
\end{equation}

which is also guaranteed to generate a sample $$\mathbf{x}_0 \sim p_0$$ whenever $$\mathbf{x}_T \sim p_T$$.

## What is Log-Density?

Diffusion models are likelihood-based models <d-cite key="song2021maximum,kingma2024understanding"></d-cite>, and as such, they aim to assign high likelihood to training data and, by extension, low likelihood to out-of-distribution (OOD) data. Intuitively, one might think that log-density is a reliable measure of whether a sample lies in or out of the data distribution.

However, prior research <d-cite key="choi2018waic,nalisnick2018deep,nalisnick2019detecting,ben2024d"></d-cite> has shown that generative models can sometimes assign higher likelihoods to OOD data than to in-distribution data. In <d-cite key="karczewski2025diffusion"></d-cite>, we show that diffusion models are no different. In fact, we push this analysis further by exploring the highest-density regions of diffusion models.

Using a theoretical **mode-tracking ODE**, we investigate the regions of the data space where the model assigns the highest likelihood. Surprisingly, these regions are occupied by cartoon-like drawings or blurry images—patterns that are absent from the training data. Additionally, we observe a strong correlation between negative log-density and PNG image size, revealing that negative log-likelihood for image data is essentially a measure of **information content** or **detail**, rather than "in-distribution-ness".

These surprising observations underscore the difference between maximum-density points and typical sets, which we explore next

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/cats_logp.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Likelihood measures the amount of detail in an image. Sample generated with a StableDiffusion v2.1 <d-cite key="rombach2021highresolution"></d-cite> using the High-Density sampler from <d-cite key="karczewski2025diffusion"></d-cite> </figcaption>
</div>

### Why Does This Happen?

The observation that blurry images or cartoon-like drawings have the highest density in diffusion models may seem counterintuitive. Does this imply that the model considers these images to be the "most likely"? To understand this phenomenon, it is crucial to distinguish between **probability density** and **probability**.

#### Probability Density vs. Probability

The probability of being in a region $$ A $$ is given by the integral of the density over that region:

$$
P(A) = \int_{A} p(\mathbf{x}) d\mathbf{x}.
$$

If the density is constant within $$ A $$, then the probability equals the product of the density and the volume of $$A $$: 

$$
P(A) = c \cdot \text{Vol}(A).
$$

**It is both the density and the volume that determine probability.**

#### A Gaussian example

A helpful analogy is the standard normal Gaussian distribution in high dimensions. Its density is proportional to $$\exp(-\|\mathbf{x}\|^2 / 2) $$ at any $$\mathbf{x} \in \mathbb{R}^D$$, which is the highest at the origin (zero vector).
However, the origin is actually far from the "typical region" of the distribution. Most samples from a high-dimensional Gaussian are not concentrated near the origin but instead fall in a region at a certain distance from it. Why is that?

If we compute the probability of a thin spherical shell (where the Gaussian density is constant) at radius $$ r $$ and thickness $$ dr $$, the volume of this shell is proportional to $$ r^{D-1}dr $$, and the probability is given by:

$$
P(\text{shell at } r) \propto r^{D-1} \exp(-r^2 / 2)dr.
$$

The key insight is that this probability is maximized not at $$ r = 0 $$ (the origin, where density is highest), but at $$ r = \sqrt{D-1} $$. <d-footnote> This is because for \( f(r)= r^{D-1} \exp(-r^2 / 2)\) we have \( f'(r)= r^{D-2} \exp(-r^2/2) (D-1 - r^2) \), and \( f'(r) > 0 \) for \( r < \sqrt{D-1} \) and \( f'(r) < 0 \) for \( r > \sqrt{D-1} \). </d-footnote>The typical region is the sweet spot, where neither the volume nor the density is too low.

#### Diffusion Models: High-Density Blurry Images vs. High-Volume Detailed Images

A similar principle applies to diffusion models. Although blurry or cartoon-like images occupy regions of high density, the "volume" of such images—i.e., the diversity of possible variations—is much smaller compared to the volume of regions corresponding to detailed, textured images. As a result, diffusion models assign lower log-densities to more detailed images.

## How to estimate Log-Density?

To measure log-density of samples from diffusion models, it’s important to understand different modes of sampling. Broadly, sampling in diffusion models can be categorized across two dimensions:

1. **Deterministic vs Stochastic Sampling**:
   - Deterministic: smooth trajectories given by ODEs.
   - Stochastic: noisy trajecotories given by SDEs.

2. **Original Dynamics vs Modified Dynamics**:
   - Following original dynamics dictated by \eqref{eq:rev-sde} or \eqref{eq:pf-ode}.
   - Following some arbitrary dynamics.

By ‘original dynamics,’ we mean the (reverse) SDE or ODE that exactly inverts the forward noising process, using the true or approximated score. By ‘any dynamics’, we allow modifications to the drift or diffusion terms, for instance (as we will see later) to control log-density.

We summarize these in the table below:

| Sampling Mode     | Original Dynamics       | Any Dynamics       |
|--------------------|-------------------------|--------------------------|
| **Deterministic** | Prior work <d-cite key="chen2018neural"></d-cite>  | Ours <d-cite key="karczewski2025diffusion"></d-cite>     |
| **Stochastic**    | Ours <d-cite key="karczewski2025diffusion"></d-cite>    | Ours <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>     |

Previously, log-density was only measurable for deterministic sampling with original dynamics. In <d-cite key="karczewski2025diffusion"></d-cite>, we extend this to deterministic sampling under modified dynamics and stochastic sampling under original dynamics.<d-footnote> Interestingly, we show in <d-cite key="karczewski2025diffusion"></d-cite> that once the true score function is replaced with the approximate one, the log-density estimate becomes biased. We derive the exact formula for this bias and show that it goes to zero when the score function estimation error does.</d-footnote>
In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we further generalize this to stochastic sampling with modified dynamics, deriving the evolution of log-density using the general **Itô's Lemma** and the **Fokker-Planck equation**.
One can see that since stochastic trajectories are a strict generalization of determinic ones (vanishing diffusion term), the method for log-density estimation for any stochastic trajectory is a strict generalization of all the other ones.

In the next section we show that the ability to estimate log-density under any dynamics allows for controlling it.

## How to Control Log-Density?

An interesting observation <d-cite key="song2021scorebased"></d-cite> is that simply rescaling the latent code (e.g., scaling the noise at the start of the sampling process) changes the amount of detail in the generated image. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we provide a theoretical explanation for this phenomenon using a concept we call Score Alignment, which directly ties the scaling of the latent code to changes in log-density.

#### Score Alignment

Score alignment measures the angle between the score function at $$t = T$$ (the noise distribution) pushed forward via PF-ODE \eqref{eq:pf-ode} to $$t = 0$$, and the score function at $$t = 0$$ (the data distribution). If the angle is always acute, scaling the latent code at $$t = T$$ changes $$\log p_0(\mathbf{x}_0)$$ in a monotonic way, explaining the relationship between scaling and image detail.<d-footnote> If the angle is always obtuse, scaling has a reverse effect, i.e. increasing \( \log p_T(\mathbf{x}_T) \) decreases \( \log p_0(\mathbf{x}_0) \) </d-footnote> Remarkably, we show that this alignment can be measured without explicitly knowing the score function, see <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> for the proof and JAX code.

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

where $$b_t$$ is a predefined function specifying how the log-density should evolve. We show that, for any flow model $$d \mathbf{x}_t=\mathbf{u}_t(\mathbf{x}_t)dt $$, which defines marginal distributions $$ p_t $$, we can modify the vector field to achieve \eqref{eq:logp-b} as long as we know the score $$ \nabla \log p_t(\mathbf{x})$$:

$$
\tilde{\mathbf{u}}_t(\mathbf{x})=\mathbf{u}_t(\mathbf{x}) + \underbrace{\frac{\operatorname{div}\mathbf{u}_t(\mathbf{x}) + b_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|^2}\nabla \log p_t(\mathbf{x})}_{\text{log-density correction}}.
$$
 
Our method minimizes deviations from the original sampling trajectory while ensuring the desired log-density changes.
The remaining question is: *How to chose $$b_t$$?*

#### Choosing Dynamics

While density guidance theoretically allows arbitrary changes to log-density, practical constraints must be considered. Log-density changes that are too large or too small can lead to samples falling outside the typical regions of the data distribution. To address this, we leverage an observation that the following term:

$$
h_t(\mathbf{x}) = \frac{\sigma_t^2 \left(\Delta \log p_t(\mathbf{x}) + \|\nabla \log p_t(\mathbf{x})\|^2\right)}{\sqrt{2D}}
$$

is approximately $$\mathcal{N}(0, 1)$$ for $$\mathbf{x} \sim p_t$$, where the data dimension $$D$$ is high. This helps determine the "typical" range of log-density changes.

\begin{equation}\label{eq:dgs}
\vu^\textsc{dg-ode}_t(\vx) = f(t)\vx - \frac{1}{2}g^2(t)\eta_t(\vx)\nabla \log p_t(\vx),
\end{equation}

which matches the PF-ODE \eqref{eq:pf-ode} with a rescaled score function by

\begin{equation}\label{eq:quantile-score-scaling}
\eta_t(\vx)=1 + \frac{\sqrt{2D}\Phi^{-1}(q)}{\| \sigma_t \nabla \log p_t(\vx) \|^2},
\end{equation}

where $$\Phi^1(q)$$ is the $$q$$-th quantile of the standard normal distribution.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/deterministic-steering.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> **Density guidance enables control over image detail.** Samples geneared with the pretrained EDM2 model <d-cite key="karras2024analyzing"></d-cite> using \eqref{eq:quantile-score-scaling} with different values of \(q\) </figcaption>
</div>

### Stochastic Sampling with Density Guidance

So far, we’ve discussed controlling log-density in deterministic settings. However, stochastic sampling introduces additional challenges and opportunities. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we extend density guidance to stochastic dynamics, showing that log-density can evolve smoothly under predefined trajectories, even when noise is injected.

\begin{equation}\label{eq:stochastic-steering}
d \vx_t =\vu_t^\textsc{dg-sde}(\vx_t)dt + \varphi(t)P_t(\vx_t)d\overline{\rW}_t
\end{equation}

\begin{equation}
\vu^\textsc{dg-sde}_t(\vx) = \vu^\textsc{dg-ode}_t(\vx)
  + \underbrace{\frac{1}{2}\varphi^2(t)\frac{\Delta \log p_t(\vx)}{\| \nabla \log p_t(\vx) \|^2}\nabla \log p_t(\vx)}_{\text{correction for added stochasticity}}
\end{equation}

\begin{equation}
P_t(\vx) = \mI_D - \left(\frac{\nabla \log p_t(\vx)}{\| \nabla \log p_t(\vx) \|}\right) \hspace{-1mm} \left(\frac{\nabla \log p_t(\vx)}{\| \nabla \log p_t(\vx) \|}\right)^T
\end{equation}

This is particularly useful for balancing detail and variability in generated samples. For example:
- Adding noise early in the sampling process introduces variation in high-level features like shapes.
- Adding noise later affects only low-level details like texture.

Our method ensures that the log-density evolution remains consistent with the deterministic case, allowing precise control while injecting controlled randomness.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/stochastic-steering.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> **Stochastic Density guidance allows for noise injection without sacrificing control over image detail.** Samples geneared with the pretrained EDM2 model <d-cite key="karras2024analyzing"></d-cite> using \eqref{eq:stochastic-steering} with different values of \(q\) and noisy injected at different stages of sampling. </figcaption>
</div>

## Conclusion

Log-density is a crucial concept in understanding and controlling diffusion models. It measures the level of detail in generated images rather than merely determining in-distribution likelihood. In <d-cite key="karczewski2025diffusion"></d-cite>, we explore the curious behavior of high-density regions in diffusion models, revealing unexpected patterns and proposing new ways to measure log-density across sampling methods. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we go further, introducing **Density Guidance** and demonstrating how to precisely control log-density in both deterministic and stochastic settings.

These findings not only advance our theoretical understanding of diffusion models but also open up practical avenues for generating images with fine-grained control over detail and variability.

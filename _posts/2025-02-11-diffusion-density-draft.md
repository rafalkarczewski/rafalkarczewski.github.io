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

The idea of diffusion models <d-cite key="sohl2015deep,ho2020denoising,song2021scorebased"></d-cite> is to gradually transform the data distribution \(p_0\) into pure noise \(p_T\) (e.g. \(\mathcal{N}(0, I)\)). This is achieved via the forward noising kernel 
\[
p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 I),
\]
with \(\alpha,\sigma\) chosen so that all the information is lost at \(t=T\), i.e. 
\[
p_T(\mathbf{x}_T \mid \mathbf{x}_0) \approx p_T(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \sigma_T^2 I).
\]
Hence, as \(t\) increases, \(\mathbf{x}_t\) becomes more “noisy,” and at \(t=T\) we reach a tractable distribution \(p_T\).

This process can equivalently be written as a Stochastic Differential Equation (SDE):

\[
d\mathbf{x}_t = f(t)\mathbf{x}_t\, dt + g(t)\,d W_t, 
\]

where \(f, g\) are scalar functions and \(W\) is the Wiener process. Remarkably, this process is reversible! The Reverse SDE <d-cite key="anderson1982reverse"></d-cite> is given by

\[
\label{eq:rev-sde}
d\mathbf{x}_t 
= \bigl(f(t)\mathbf{x}_t - g^2(t)\nabla \log p_t(\mathbf{x}_t)\bigr)\,dt
+ g(t)\,d \overline{W}_t,
\]

where \(\overline{W}\) is the Wiener process going backwards in time and \(\nabla \log p_t(\mathbf{x}_t)\) is the *score function*, which can be accurately approximated with a neural network <d-cite key="hyvarinen2005estimation,vincent2011connection,song2020sliced"></d-cite>. Since \(p_T\) is a tractable distribution, we can easily sample \(\mathbf{x}_T \sim p_T\) and solve \eqref{eq:rev-sde} to generate a sample \(\mathbf{x}_0 \sim p_0\).

Rather surprisingly, it turns out that there exists an equivalent *deterministic* process <d-cite key="song2021scorebased,song2020denoising"></d-cite> given by an Ordinary Differential Equation (ODE):

\[
\label{eq:pf-ode}
d\mathbf{x}_t 
= \Bigl(f(t)\mathbf{x}_t - \tfrac{1}{2}g^2(t)\nabla \log p_t(\mathbf{x}_t)\Bigr)\,dt,
\]

which is also guaranteed to generate a sample \(\mathbf{x}_0 \sim p_0\) whenever \(\mathbf{x}_T \sim p_T\). 

---

## What is Log-Density?

Diffusion models are likelihood-based models <d-cite key="song2021maximum,kingma2024understanding"></d-cite>, aiming to assign high likelihood to training data and, by extension, lower likelihood to out-of-distribution (OOD) data. Intuitively, one might think that log-density is a reliable measure of whether a sample is “in” or “out” of the data distribution.

However, prior research <d-cite key="choi2018waic,nalisnick2018deep,nalisnick2019detecting,ben2024d"></d-cite> has shown that generative models can sometimes assign higher likelihoods to OOD data than to in-distribution data. In <d-cite key="karczewski2025diffusion"></d-cite>, we show that diffusion models are no different. In fact, we push this analysis further by exploring the *highest-density* regions of diffusion models.

Using a theoretical **mode-tracking ODE**, we investigate which parts of the data space the model deems highest likelihood. Surprisingly, these regions turn out to be cartoon-like drawings or blurry images—patterns absent from the training data. Moreover, we observe a strong correlation between negative log-density and PNG image size, indicating that negative log-likelihood for image data primarily reflects **information content** or **detail**, rather than “in-distribution-ness.”

These surprising observations underscore the difference between maximum-density points and *typical* sets, which we explore next.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/cats_logp.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Likelihood measures the amount of detail in an image. Sample generated with StableDiffusion v2.1 <d-cite key="rombach2021highresolution"></d-cite> using the High-Density sampler from <d-cite key="karczewski2025diffusion"></d-cite>. </figcaption>
</div>

### Why Does This Happen?

The observation that blurry or cartoon-like images have the highest density may seem counterintuitive. Does it imply the model considers these images to be the “most likely”? To clarify this, we must distinguish between **probability density** and **probability**.

#### Probability Density vs. Probability

The probability of a region \(A\) is given by:

\[
P(A) = \int_{A} p(\mathbf{x})\, d\mathbf{x}.
\]

If the density \(p(\mathbf{x})\) is constant over \(A\), then

\[
P(A) = \text{(constant density)} \times \text{Vol}(A).
\]

Hence, *both* density and volume determine the probability.

#### A Gaussian Example

Consider a standard normal Gaussian in high dimensions. Its density is proportional to \(\exp(-\|\mathbf{x}\|^2 / 2)\), which peaks at the origin \(\mathbf{x} = \mathbf{0}\). However, typical samples from a high-dimensional Gaussian lie far from the origin. 

In fact, if you look at a thin spherical shell of radius \(r\) (where the density is constant) with thickness \(dr\), its volume is proportional to \(r^{D-1} dr\). The probability of being in this shell is:

\[
P(\text{shell at }r) 
\;\propto\; r^{D-1}\,\exp\bigl(-r^2/2\bigr)\,dr.
\]

This is maximized at \(r = \sqrt{D-1}\).<d-footnote>Because \( f(r)= r^{D-1} \exp(-r^2/2)\) satisfies \( f'(r)= r^{D-2} \exp(-r^2/2) (D-1 - r^2)\). Thus \(f'(r)>0\) for \(r<\sqrt{D-1}\) and \(f'(r)<0\) for \(r>\sqrt{D-1}\).</d-footnote> The origin (\(r=0\)) has the *highest density* but negligible volume, so it’s not in the *typical region*.

#### Diffusion Models: High-Density Blurry Images vs. High-Volume Detailed Images

By analogy, blurry/cartoon-like images correspond to high-density but small-volume regions, whereas detailed images occupy a larger-volume region with lower density. This explains why typical real-like images can have lower log-density than simpler, less-detailed images.

---

## How to Estimate Log-Density?

To measure log-density of samples from diffusion models, it’s helpful to categorize sampling along two dimensions:

1. **Deterministic vs. Stochastic Sampling**  
   - Deterministic: smooth trajectories via ODEs.  
   - Stochastic: noisy trajectories via SDEs.

2. **Original Dynamics vs. Modified Dynamics**  
   - *Original Dynamics*: The reverse process in \eqref{eq:rev-sde} or \eqref{eq:pf-ode}.  
   - *Any Dynamics*: Modified drift or diffusion terms (e.g., to guide the model toward certain log-density targets).

| Sampling Mode     | Original Dynamics                                       | Any Dynamics                                          |
|-------------------|---------------------------------------------------------|-------------------------------------------------------|
| **Deterministic** | Prior work <d-cite key="chen2018neural"></d-cite>      | Ours <d-cite key="karczewski2025diffusion"></d-cite> |
| **Stochastic**    | Ours <d-cite key="karczewski2025diffusion"></d-cite>   | Ours <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>   |

Previously, log-density was only measurable for *deterministic* sampling under the *original* dynamics. In <d-cite key="karczewski2025diffusion"></d-cite>, we extend this to deterministic sampling under modified dynamics, as well as stochastic sampling under original dynamics.<d-footnote>In <d-cite key="karczewski2025diffusion"></d-cite>, we also show that if the *true* score is replaced by an *approximate* one, the log-density estimate becomes biased. We derive the exact bias formula and show it vanishes if the score approximation error goes to zero.</d-footnote> Subsequently, in <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we further generalize log-density estimation to *stochastic sampling with modified dynamics*, using Itô’s Lemma and the Fokker–Planck equation.

In practice, these results are particularly relevant for **diffusion models**, since we already have access to an approximate score function. The formulas themselves can be applied to any continuous-time flow \(\mathbf{x}_t\), but knowledge of \(\nabla \log p_t(\mathbf{x}_t)\) is crucial to making them work in practice.

---

## How to Control Log-Density?

An interesting empirical observation <d-cite key="song2021scorebased"></d-cite> is that simply **rescaling** the latent code (e.g., scaling the initial noise \(\mathbf{x}_T\)) changes the amount of detail in the generated image. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we provide a theoretical explanation of this via **Score Alignment**, which directly ties the scaling of latent noise to changes in log-density.

### Score Alignment

Score Alignment (SA) measures the angle between:

- the score function at \(t = T\) (for the noise distribution), *pushed forward* via the PF-ODE \eqref{eq:pf-ode} to \(t = 0\), and  
- the score function at \(t = 0\) (the data distribution).

If this angle is *acute*, then scaling \(\mathbf{x}_T\) at \(t=T\) *monotonically* changes \(\log p_0(\mathbf{x}_0)\).<d-footnote>If the angle is always obtuse, the relationship inverts: increasing \(\log p_T(\mathbf{x}_T)\) decreases \(\log p_0(\mathbf{x}_0)\).</d-footnote> Remarkably, SA can be checked *without* explicitly knowing the entire score, as shown in <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/sa_vis.png">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Score Alignment guarantees a monotonic relationship between scaling the latent code and the final log-density. Empirically, it nearly always holds for diffusion models on image data. </figcaption>
</div>

**Take-Home:** *If Score Alignment holds, simply rescaling \(\mathbf{x}_T\) is enough to reliably increase or decrease the final log-density (and thus image detail).*

---

### Density Guidance: A Principled Approach to Controlling Log-Density

While latent code scaling is convenient, it’s a *coarse* control. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we introduce **Density Guidance**, a more precise way to control the trajectory of \(\log p_t(\mathbf{x}_t)\) during sampling. We start from the PF-ODE \eqref{eq:pf-ode} but modify it to enforce:

\[
\label{eq:logp-b}
\frac{d \log p_t(\mathbf{x}_t)}{dt} \;=\; b_t(\mathbf{x}_t),
\]

where \(b_t\) is a user-defined function specifying how we want log-density to evolve over time.

#### Deterministic Density Guidance

To achieve \(\tfrac{d}{dt} \log p_t(\mathbf{x}_t) = b_t(\mathbf{x}_t)\) in a diffusion model (where we can approximate \(\nabla \log p_t(\mathbf{x})\)), we define a *corrected* drift field:

\[
\begin{equation}
\label{eq:modified-drift}
\tilde{\mathbf{u}}_t(\mathbf{x})
= \mathbf{u}_t(\mathbf{x})
+ \underbrace{\frac{\operatorname{div}\mathbf{u}_t(\mathbf{x}) + b_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|^2}\,\nabla \log p_t(\mathbf{x})}_{\text{log-density correction}},
\end{equation}
\]

where \(\mathbf{u}_t(\mathbf{x})\) is the original flow (e.g., from \eqref{eq:pf-ode}), and \(\operatorname{div}\) denotes divergence. We then integrate

\[
d \mathbf{x}_t = \tilde{\mathbf{u}}_t(\mathbf{x}_t)\,dt,
\]

which guides \(\mathbf{x}_t\) to follow any desired log-density schedule \(b_t\).

- **Choosing \(b_t\)**: If \(b_t\) is too large or too small, the sample may leave the typical region of the data. In high-dimensional settings, a practical choice leverages the observation that
  \[
  h_t(\mathbf{x}) 
  = \frac{\sigma_t^2 \,\bigl[\Delta \log p_t(\mathbf{x}) + \|\nabla \log p_t(\mathbf{x})\|^2\bigr]}{\sqrt{2D}}
  \]
  behaves approximately like \(\mathcal{N}(0,1)\) for \(\mathbf{x}\sim p_t\). From this, we propose a user-friendly function \(b^q_t(\mathbf{x})\) that can raise or lower \(\log p_0(\mathbf{x}_0)\) by choosing a quantile \(q\).

Concretely, the resulting **Density Guidance ODE**:

\[
\mathbf{u}^{\text{DG-ODE}}_t(\mathbf{x}) 
= f(t)\,\mathbf{x} \;-\; \tfrac{1}{2}g^2(t)\,\eta_t(\mathbf{x})\,\nabla \log p_t(\mathbf{x}),
\]

matches the original PF-ODE \eqref{eq:pf-ode} but with a *rescaled* score \(\eta_t(\mathbf{x})\,\nabla \log p_t(\mathbf{x})\). This rescaling is *adaptive* in both \(t\) and \(\mathbf{x}\), allowing fine-grained control of how \(\log p_t(\mathbf{x}_t)\) evolves. 

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/deterministic-steering.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> **Density guidance enables precise control over image detail.** Samples generated with a pretrained EDM2 model <d-cite key="karras2024analyzing"></d-cite> using different values of \(q\). Higher \(q\) raises log-density, leading to simpler or smoother images. </figcaption>
</div>

**Take-Home:** *Deterministic Density Guidance = run the PF-ODE with a *rescaled* score function. This gives finer control than basic latent code scaling.*

---

### Stochastic Sampling with Density Guidance

In many applications, we still want to inject noise for additional sample diversity, but *without* losing log-density control. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we extend density guidance to SDE sampling:

\[
\label{eq:stochastic-steering}
d \mathbf{x}_t 
= \mathbf{u}_t^{\text{DG-SDE}}(\mathbf{x}_t)\,dt 
+ \varphi(t)\,P_t(\mathbf{x}_t)\,d\overline{W}_t,
\]

where \(\mathbf{u}_t^{\text{DG-SDE}}(\mathbf{x})\) again modifies the drift to ensure \(\tfrac{d}{dt}\log p_t(\mathbf{x}_t)\) follows \(b_t\). Concretely,

\[
\mathbf{u}^{\text{DG-SDE}}_t(\mathbf{x}) 
= \mathbf{u}^{\text{DG-ODE}}_t(\mathbf{x})
\;+\; 
\underbrace{
  \tfrac{1}{2}\,\varphi^2(t)\,\frac{\Delta \log p_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|^2}\,\nabla \log p_t(\mathbf{x})
}_{\text{correction for added stochasticity}},
\]

and

\[
P_t(\mathbf{x})
= I 
 - \frac{\nabla \log p_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|} 
   \Bigl(\frac{\nabla \log p_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|}\Bigr)^T,
\]
is a projection matrix onto the subspace orthogonal to \(\nabla \log p_t(\mathbf{x})\). This ensures that, even though \(\mathbf{x}_t\) is stochastic, the change in \(\log p_t(\mathbf{x}_t)\) is *exactly* the same as in the deterministic version.

<d-footnote>
Because we project the Wiener process onto the orthogonal complement of the score, one can derive an additional small drift term. Empirically, we find this correction negligible and omit it in experiments. See <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> for details.
</d-footnote>

In practice, we set \(\varphi(t)=\widetilde{\varphi}(t)\,g(t)\) so that the noise strength is relative to the original diffusion coefficient \(g(t)\). The net effect is that we still have a *score-rescaled* drift (as in deterministic guidance) plus the ability to add controllable noise, e.g.:

- Adding noise early introduces global shape variation.
- Adding noise late affects mainly local texture.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/stochastic-steering.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> **Stochastic Density Guidance** allows noise injection without losing control over image detail. Samples generated with a pretrained EDM2 model <d-cite key="karras2024analyzing"></d-cite>, varying \(q\) and the timing of noise injection. </figcaption>
</div>

**Take-Home:** *Stochastic Density Guidance = the PF-ODE with a **rescaled** score, plus a projected noise term. This yields a smooth log-density schedule while still allowing for randomness.*

---

## Conclusion

Log-density is a crucial concept for understanding and controlling diffusion models. Rather than signaling “in-distribution-ness,” it mostly reflects the *level of detail* in generated images. In <d-cite key="karczewski2025diffusion"></d-cite>, we investigate the peculiar behavior of high-density regions in diffusion models, revealing unexpected artifacts and providing new tools for measuring log-density under various sampling regimes. Building on this, <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> introduces **Density Guidance**, a systematic framework—both deterministic and stochastic—that lets us precisely shape how \(\log p_t(\mathbf{x}_t)\) evolves during generation.

These findings deepen our theoretical grasp of diffusion models and open up **practical avenues** for generating images with fine-grained control over detail and variability.

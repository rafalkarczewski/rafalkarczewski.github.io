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

## Diffusion models recap

The idea of diffusion models <d-cite key="sohl2015deep,ho2020denoising,song2021scorebased"></d-cite> is to gradually transform the data distribution $$p_0$$ into pure noise $$p_T$$ (e.g. $$\mathcal{N}(0, I)$$). This is achieved via the forward noising kernel 
$$
p_t(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 I),
$$
with $$\alpha, \sigma$$ chosen so that all the information is lost at $$t=T$$, i.e. 
$$
p_T(\mathbf{x}_T \mid \mathbf{x}_0) \approx p_T(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \sigma_T^2 I).
$$
Hence, as $$t$$ increases, $$\mathbf{x}_t$$ becomes more "noisy," and at $$t=T$$ we reach a tractable distribution $$p_T$$.

This process can equivalently be written as a Stochastic Differential Equation (SDE):

$$
d\mathbf{x}_t = f(t)\mathbf{x}_t\, dt + g(t)\,d W_t,
$$

where $$f, g$$ are scalar functions and $$W$$ is the Wiener process. Remarkably, this process is reversible! The Reverse SDE <d-cite key="anderson1982reverse"></d-cite> is given by

$$\begin{equation}\label{eq:rev-sde}
d\mathbf{x}_t = \bigl(f(t)\mathbf{x}_t - g^2(t)\nabla \log p_t(\mathbf{x}_t)\bigr)\,dt + g(t)\,d \overline{W}_t,
\end{equation}$$

where $$\overline{W}$$ is the Wiener process going backwards in time and $$\nabla \log p_t(\mathbf{x}_t)$$ is the *score function*, which can be accurately approximated with a neural network <d-cite key="hyvarinen2005estimation,vincent2011connection,song2020sliced"></d-cite>. Since $$p_T$$ is a tractable distribution, we can easily sample $$\mathbf{x}_T \sim p_T$$ and solve \eqref{eq:rev-sde} to generate a sample $$\mathbf{x}_0 \sim p_0$$.

Rather surprisingly, there exists an equivalent *deterministic* process <d-cite key="song2021scorebased,song2020denoising"></d-cite> given by an Ordinary Differential Equation (ODE):

$$\begin{equation}\label{eq:pf-ode}
d\mathbf{x}_t = \Bigl(f(t)\mathbf{x}_t - \frac{1}{2}g^2(t)\nabla \log p_t(\mathbf{x}_t)\Bigr)\, dt,
\end{equation}$$

which is also guaranteed to generate a sample $$\mathbf{x}_0 \sim p_0$$ whenever $$\mathbf{x}_T \sim p_T$$.

---

## What is Log-Density?

Diffusion models are likelihood-based models <d-cite key="song2021maximum,kingma2024understanding"></d-cite>, aiming to assign high likelihood to training data and, by extension, lower likelihood to out-of-distribution (OOD) data. Intuitively, one might think that log-density is a reliable measure of whether a sample is “in” or “out” of the data distribution.

However, prior research <d-cite key="choi2018waic,nalisnick2018deep,nalisnick2019detecting,ben2024d"></d-cite> has shown that generative models can sometimes assign higher likelihoods to OOD data than to in-distribution data. In <d-cite key="karczewski2025diffusion"></d-cite>, we show that diffusion models are no different. In fact, we push this analysis further by exploring the *highest-density* regions of diffusion models.

Using a theoretical **mode-tracking ODE**, we investigate the regions of the data space where the model assigns the highest likelihood. Surprisingly, these regions are occupied by cartoon-like drawings or blurry images—patterns absent from the training data. Additionally, we observe a strong correlation between negative log-density and PNG image size, revealing that negative log-likelihood for image data essentially measures **information content** (or **detail**), rather than “in-distribution-ness.”

These surprising observations underscore the difference between maximum-density points and *typical* sets, which we explore next.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/cats_logp.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
 Likelihood measures the amount of detail in an image. Sample generated with a StableDiffusion v2.1 <d-cite key="rombach2021highresolution"></d-cite> using the High-Density sampler from <d-cite key="karczewski2025diffusion"></d-cite>.
</figcaption>
</div>

### Why Does This Happen?

The observation that blurry or cartoon-like images have the highest density may seem counterintuitive. Does it imply the model considers these images to be the “most likely”? To clarify this, we must distinguish between **probability density** and **probability**.

#### Probability Density vs. Probability

The probability of being in a region $$ A $$ is given by the integral of the density over that region:

$$
P(A) = \int_A p(\mathbf{x})\, d\mathbf{x}.
$$

If the density $$p(\mathbf{x})$$ is constant in $$A$$, then

$$
P(A) = \text{(constant density)} \times \text{Vol}(A).
$$

Hence, *both* density and volume determine the probability.

#### A Gaussian Example

A helpful analogy is the standard normal Gaussian in high dimensions. Its density is proportional to $$\exp(-\|\mathbf{x}\|^2/2)$$, which is highest at the origin $$\mathbf{x}=\mathbf{0}$$. However, typical samples in high dimensions lie *away* from the origin.

Specifically, consider a thin spherical shell of radius $$r$$ and thickness $$dr$$. The volume is proportional to $$r^{D-1}dr$$, and the probability is

$$
P(\text{shell at }r) \;\propto\; r^{D-1}\,\exp\Bigl(-\frac{r^2}{2}\Bigr)\,dr.
$$

This is maximized at $$r=\sqrt{D-1}$$. <d-footnote>This is because \(f(r)= r^{D-1} \exp(-r^2/2)\) satisfies \(f'(r)= r^{D-2} \exp(-r^2/2) (D-1 - r^2)\), which is zero at \(r=\sqrt{D-1}\). </d-footnote> Thus, although density is highest at the origin, that point has negligible *volume* and is not in the typical region.

#### Diffusion Models: High-Density Blurry Images vs. High-Volume Detailed Images

By analogy, blurry/cartoon-like images correspond to small-volume, high-density regions, while detailed images occupy a larger-volume region with lower density. Hence, more realistic images can have *lower* log-density than simpler, less-detailed ones.

---

## How to Estimate Log-Density?

To measure log-density of diffusion samples, note that sampling can vary along two dimensions:

1. **Deterministic vs. Stochastic Sampling**  
   - Deterministic: smooth trajectories (ODE-based).  
   - Stochastic: noisy trajectories (SDE-based).

2. **Original Dynamics vs. Modified Dynamics**  
   - *Original Dynamics*: The reverse process given by \eqref{eq:rev-sde} or \eqref{eq:pf-ode}.  
   - *Any Dynamics*: Modified drift/diffusion to target a specific objective (e.g., controlling log-density).

We can summarize:

| Sampling Mode     | Original Dynamics                                       | Any Dynamics                                                |
|-------------------|---------------------------------------------------------|-------------------------------------------------------------|
| **Deterministic** | Prior work <d-cite key="chen2018neural"></d-cite>      | Ours <d-cite key="karczewski2025diffusion"></d-cite>       |
| **Stochastic**    | Ours <d-cite key="karczewski2025diffusion"></d-cite>   | Ours <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> |

Previously, log-density was only measurable for *deterministic* sampling under the *original* dynamics. In <d-cite key="karczewski2025diffusion"></d-cite>, we extend this to deterministic sampling with modified dynamics, as well as stochastic sampling under original dynamics.<d-footnote>Once the *true* score is replaced by an approximate one, log-density estimates become biased. We derive the exact bias in <d-cite key="karczewski2025diffusion"></d-cite> and show it vanishes as the approximation error goes to zero.</d-footnote>

In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we further generalize to *stochastic sampling with modified dynamics*, deriving the log-density evolution via **Itô's Lemma** and the **Fokker–Planck equation**. In practice, these formulas are most relevant to diffusion models because we already have (an approximation of) $$\nabla \log p_t(\mathbf{x})$$. The same framework can be used for *any* continuous-time flow model, provided the score is known.

---

## How to Control Log-Density?

An interesting empirical observation <d-cite key="song2021scorebased"></d-cite> is that simply **rescaling** the latent code (e.g., scaling the initial noise at $$t=T$$) can change the level of detail in the generated image. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we offer a theoretical explanation using the concept of **Score Alignment**, which relates scaling the latent code to changes in log-density.

### Score Alignment

**Score Alignment (SA)** measures the angle between

- the score function at $$t=T$$ (noise distribution), *pushed forward* to $$t=0$$ via \eqref{eq:pf-ode}, and  
- the score function at $$t=0$$ (data distribution).

If this angle is always *acute*, increasing $$\log p_T(\mathbf{x}_T)$$ (e.g., by scaling $$\mathbf{x}_T$$) *monotonically* increases $$\log p_0(\mathbf{x}_0)$$.<d-footnote>If it is always obtuse, increasing \( \log p_T(\mathbf{x}_T)\) decreases \( \log p_0(\mathbf{x}_0)\). </d-footnote> Surprisingly, one can verify this alignment *without* fully knowing the score function; see <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> for details.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/sa_vis.png">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
 Score Alignment guarantees a monotonic relationship between scaling the latent code at \(t=T\) and the resulting log-density at \(t=0\). Empirically, it nearly always holds for diffusion models on image data.
</figcaption>
</div>

**Take-home:** *If SA holds, simply rescaling the latent noise \(\mathbf{x}_T\) is a quick way to increase or decrease the final log-density (and thus control image detail).*

---

### Density Guidance: A Principled Approach to Controlling Log-Density

While latent code scaling works, it is fairly coarse. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we propose **Density Guidance**, a more precise way to steer how $$\log p_t(\mathbf{x}_t)$$ evolves during sampling. We start from a general flow model

$$
d\mathbf{x}_t = \mathbf{u}_t(\mathbf{x}_t)\,dt,
$$

and want to enforce

$$\begin{equation}\label{eq:logp-b}
\frac{d}{dt}\,\log p_t(\mathbf{x}_t) = b_t(\mathbf{x}_t),
\end{equation}$$

for a user-defined function $$b_t(\mathbf{x}_t)$$. As long as we know the score $$\nabla \log p_t(\mathbf{x})$$, we can modify $$\mathbf{u}_t(\mathbf{x})$$:

$$\begin{equation}
\tilde{\mathbf{u}}_t(\mathbf{x})
= \mathbf{u}_t(\mathbf{x})
+ \underbrace{\frac{\operatorname{div}\,\mathbf{u}_t(\mathbf{x}) + b_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|^2}\,\nabla \log p_t(\mathbf{x})}_{\text{log-density correction}}.
\end{equation}$$

#### Choosing Dynamics

In diffusion models, we typically let $$\mathbf{u}_t(\mathbf{x})$$ be given by the PF-ODE \eqref{eq:pf-ode}, so the original flow is

$$
\mathbf{u}_t(\mathbf{x}) 
= f(t)\,\mathbf{x} - \tfrac{1}{2}g^2(t)\,\nabla \log p_t(\mathbf{x}).
$$

If $$b_t(\mathbf{x})$$ is too large, samples may leave the typical region; if it is too small, we might not observe a noticeable density shift. A practical strategy is to choose $$b_t(\mathbf{x})$$ based on typical fluctuations of $$\log p_t(\mathbf{x})$$ in high dimensions (e.g., the $$\mathcal{N}(0,1)$$-like behavior of certain score-based terms). One concrete example is:

$$\begin{equation}\label{eq:b-quantile}
b^q_t(\mathbf{x}) 
= -\operatorname{div}\,\mathbf{u}_t(\mathbf{x})
 \;-\; \frac{1}{2}\,g^2(t)\,\frac{\sqrt{2D}}{\sigma_t^2}\,\Phi^{-1}(q),
\end{equation}\end{equation}$$

leading to a **Density Guidance ODE**:

$$
\mathbf{u}_t^{\text{DG-ODE}}(\mathbf{x})
= f(t)\,\mathbf{x}
 - \tfrac{1}{2}g^2(t)\,\eta_t(\mathbf{x})\,\nabla \log p_t(\mathbf{x}),
$$

where $$\eta_t(\mathbf{x})$$ adaptively rescales the score.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/deterministic-steering.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
 **Density Guidance** in a deterministic setting. Using \eqref{eq:b-quantile} with different \(q\) yields varying amounts of detail in samples from a pretrained EDM2 model <d-cite key="karras2024analyzing"></d-cite>.
</figcaption>
</div>

**Take-home:** *Deterministic Density Guidance modifies the PF-ODE with a **rescaled** score, achieving fine-grained control of log-density over the entire sampling trajectory.*

---

### Stochastic Sampling with Density Guidance

Many applications benefit from injecting noise for diversity but still require log-density control. In <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite>, we introduce a stochastic variant:

$$\begin{equation}\label{eq:stochastic-steering}
d \mathbf{x}_t 
= \mathbf{u}_t^{\text{DG-SDE}}(\mathbf{x}_t)\,dt 
+ \varphi(t)\,P_t(\mathbf{x}_t)\,d\overline{W}_t,
\end{equation}\end{equation}$$

with

$$
\mathbf{u}_t^{\text{DG-SDE}}(\mathbf{x}) 
= \mathbf{u}_t^{\text{DG-ODE}}(\mathbf{x})
 + \underbrace{\frac{1}{2}\,\varphi^2(t)\,\frac{\Delta \log p_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|^2}\,\nabla \log p_t(\mathbf{x})}_{\text{correction for added stochasticity}},
$$

and

$$
P_t(\mathbf{x})
= I 
 - \frac{\nabla \log p_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|}\,\Bigl(\frac{\nabla \log p_t(\mathbf{x})}{\|\nabla \log p_t(\mathbf{x})\|}\Bigr)^T.
$$

The matrix $$P_t(\mathbf{x})$$ projects the Wiener increment onto the subspace orthogonal to the score. This ensures $$\log p_t(\mathbf{x}_t)$$ evolves just like in the deterministic case, *even though* $$\mathbf{x}_t$$ itself follows a stochastic path.

<d-footnote>
Strictly, projecting out the score direction also introduces a small extra drift term. Empirically, this term is negligible, so we omit it in experiments. See <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> for details.
</d-footnote>

In practice, we let $$\varphi(t)=\widetilde{\varphi}(t)\,g(t)$$, controlling how much extra noise we add relative to the original SDE’s diffusion coefficient $$g(t)$$. This yields a **Stochastic Density Guidance** method that smoothly steers $$\log p_t(\mathbf{x}_t)$$ while permitting randomness in orthogonal directions. For example:

- Early noise injection changes large-scale shapes or features.
- Late noise injection changes only fine textures.

<div class='l-body'>
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/density-guidance/stochastic-steering.jpg">
<figcaption class="figcaption" style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
 **Stochastic Density Guidance** adds noise without forfeiting log-density control. Samples from a pretrained EDM2 model <d-cite key="karras2024analyzing"></d-cite>, varying \(q\) and the timing of noise injection.
</figcaption>
</div>

**Take-home:** *Stochastic Density Guidance = same **rescaled** score approach, plus a projected noise term that preserves the intended log-density schedule.*

---

## Conclusion

Log-density is central to understanding and controlling diffusion models. Rather than signifying “in-distribution” membership, it mostly reflects the *amount of detail* in generated images. In <d-cite key="karczewski2025diffusion"></d-cite>, we investigate the peculiar high-density regions of diffusion models, revealing surprising artifacts and offering new ways to measure log-density in diverse sampling regimes. Building on this, <d-cite key="karczewski2025devildetailsdensityguidance"></d-cite> introduces **Density Guidance**, a framework—both deterministic and stochastic—that lets us precisely sculpt how $$\log p_t(\mathbf{x}_t)$$ evolves during sampling.

These findings deepen our theoretical grasp of diffusion models and open up **practical avenues** for generating images with fine-grained control over detail and variability.


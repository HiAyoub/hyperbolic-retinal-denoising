# References

Annotated bibliography of papers and resources directly relevant to this project, grouped by role. For each entry, the brief comment explains *why* it matters for our work, not just what it is.

## Base paper this project adapts

**Mishra, Shreya S., van Spengler, Max, Berkhout, Erwin & Mettes, Pascal (2026).** *Hyperbolic U-Net for Robust Medical Image Segmentation.* Medical Imaging with Deep Learning (MIDL). [[OpenReview](https://openreview.net/forum?id=NxKaeTNMxR)] [[Code](https://github.com/swastishreya/Hyperbolic-U-Net)]

> The MIDL 2026 paper this internship adapts. Shows that a fully hyperbolic U-Net (Poincaré ball, hypll library) is more robust to image noise than its Euclidean twin on medical segmentation benchmarks including REFUGE2 (fundus). Our hypothesis is that this robustness benefit transfers from segmentation to restoration. Both Euclidean and Hyperbolic variants ship in the same codebase, which gave us the apples-to-apples comparison framework we adopted.

## Tools and libraries

**van Spengler, Max, Wirth, Philipp & Mettes, Pascal (2023).** *HypLL: The Hyperbolic Learning Library.* arXiv:2306.06154. [[arXiv](https://arxiv.org/abs/2306.06154)] [[GitHub](https://github.com/maxvanspengler/hyperbolic_learning_library)]

> The `hypll` PyTorch extension. Provides `PoincareBall`, `Curvature`, `HConvolution2d`, `HMaxPool2d`, `HReLU`, `HLinear`, `TangentTensor`, `expmap` / `logmap`, and `RiemannianAdam`. All hyperbolic operations in our notebooks come from here. Well-tested, MIT-licensed, maintained by one of the Mishra et al. co-authors.

## Foundational hyperbolic-DL literature

**Nickel, Maximilian & Kiela, Douwe (2017).** *Poincaré Embeddings for Learning Hierarchical Representations.* NeurIPS.

> The original deep-learning demonstration that hyperbolic embeddings of trees beat Euclidean ones at any finite dimension. Key motivation for why we expect retinal vessels (a tree) to benefit from a hyperbolic representation.

**Nickel, Maximilian & Kiela, Douwe (2018).** *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry.* ICML.

> The follow-up arguing the Lorentz (hyperboloid) model is numerically more stable than the Poincaré ball, especially for deep networks. Cited in this project as the basis for any future Lorentz comparison experiment if Poincaré numerical issues surface.

**Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I. & Lempitsky, V. (2020).** *Hyperbolic Image Embeddings.* CVPR.

> Early demonstration that hyperbolic image embeddings (single hyperbolic layer atop a Euclidean CNN backbone) help classification on certain hierarchically structured datasets. Closest prior art for the partial-hyperbolic-intervention design pattern we tested in Phases 3 / 3.1.

**Atigh, M. G., Schoep, J., Acar, E., van Noord, N. & Mettes, P. (2022).** *Hyperbolic Image Segmentation.* CVPR.

> The conceptual predecessor of Mishra et al. — applies hyperbolic geometry to dense prediction (segmentation) rather than classification. Shows that hyperbolic logits help with class hierarchies in segmentation.

## Image denoising

**Zhang, K., Zuo, W., Chen, Y., Meng, D. & Zhang, L. (2017).** *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.* IEEE Transactions on Image Processing (DnCNN).

> Introduces residual denoising — the network predicts the noise rather than the clean image. We adopt this pattern for our output head (returns `noisy − noise_pred`). The four-argument defense in `docs/phase2_design_rationale.pdf` §5 cites this paper.

**Zhao, H., Gallo, O., Frosio, I. & Kautz, J. (2017).** *Loss Functions for Image Restoration with Neural Networks.* IEEE Transactions on Computational Imaging.

> The empirical study comparing L1, L2, SSIM, and MS-SSIM losses for image restoration. Their finding that L1 produces sharper outputs than L2 is the basis for our loss choice.

**Foi, A., Trimeche, M., Katkovnik, V. & Egiazarian, K. (2008).** *Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-image Raw-data.* IEEE Transactions on Image Processing.

> Standard reference for the Poisson-Gaussian noise model `var(y) = α·μ + σ²` used throughout this project (Phase 1 noise characterization + Phase 2 / 3 / 4 training).

**Cherukuri, V., Kanike, M. R., Mounika, T. & Gandhi, V. (2020).** *Deep Retinal Image Restoration with Generative Adversarial Networks.* IEEE Transactions on Medical Imaging.

> One of the few prior works on retinal denoising specifically. Useful as a sanity benchmark for the absolute PSNR/SSIM numbers our Euclidean baseline reaches.

## Loss function alternatives (for future work)

**Johnson, J., Alahi, A. & Fei-Fei, L. (2016).** *Perceptual Losses for Real-Time Style Transfer and Super-Resolution.* ECCV.

> The canonical perceptual-loss paper. Proposes using pretrained VGG features as a distance metric, in addition to or instead of pixel L1/L2. If we end up adding a perceptual term to address the vessel-softening issue, this is the citation.

**Ledig, C. et al. (2017).** *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN).* CVPR.

> Combines perceptual loss with adversarial training. Reframes "pixel loss → blurry output" as a fundamental limitation of L2/L1 rather than an implementation defect. Frames the failure mode we observe (vessel softening) in standard image-restoration terms.

**Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. (2018).** *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS).* CVPR.

> Defines LPIPS, a learned perceptual distance that correlates with human perceptual judgments far better than PSNR or SSIM. If we add a third metric to the eval, it should be LPIPS.

## Optimization

**Loshchilov, I. & Hutter, F. (2017).** *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR.

> Introduces cosine annealing of the learning rate. Our training recipe uses cosine annealing to zero with a one-epoch linear warmup. The Phase 2 design rationale §8 cites this paper.

**Loshchilov, I. & Hutter, F. (2019).** *Decoupled Weight Decay Regularization (AdamW).* ICLR.

> AdamW separates weight decay from the gradient step. We use AdamW for the Euclidean baseline (Phases 1 / 2 / 2.1) and `RiemannianAdam` for the hyperbolic ones (Phases 3 / 3.1 / 4).

**Bécigneul, G. & Ganea, O.-E. (2019).** *Riemannian Adaptive Optimization Methods.* ICLR.

> Theoretical basis for `RiemannianAdam`. Defines Adam-style adaptive updates that respect manifold structure (parameters constrained to a Riemannian manifold should not take Euclidean steps).

## Dataset

**Pachade, S. et al. (2021).** *Retinal Fundus Multi-disease Image Dataset (RFMiD): A Dataset for Multi-disease Detection Research.* Data, MDPI.

> The RFMiD dataset paper. 3,200 fundus photographs with 28 disease labels, split into official train (1920), validation (640), and test (640) sets. We use only the images (no labels) for our denoising task. Adopted as our dataset because (a) fundus images are clinically meaningful, (b) it has an active research community, (c) it ships official splits enabling literature comparison.

## Background / textbooks

**Cannon, J. W., Floyd, W. J., Kenyon, R. & Parry, W. R. (1997).** *Hyperbolic Geometry.* Flavors of Geometry, MSRI Publications.

> Mathematical primer on hyperbolic geometry, the Poincaré ball model, and the relationship between metric, geodesics, and the Möbius-style group structure. Useful for anyone reviewing the project who hasn't seen non-Euclidean geometry recently.

**Ungar, A. A. (2008).** *A Gyrovector Space Approach to Hyperbolic Geometry.* Synthesis Lectures on Mathematics & Statistics, Morgan & Claypool.

> The "gyrovector" formalism for hyperbolic operations (Möbius addition as a non-associative analogue of vector addition). This is the algebraic structure `hypll` implements internally.

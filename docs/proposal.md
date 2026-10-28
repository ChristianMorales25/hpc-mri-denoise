# Project proposal — CIIC 5019

**Title:** GPU-Accelerated 3D MRI Denoising and Its Impact on Downstream Tissue Segmentation

**Student:** Christian Morales (solo)

**Project from the list:** #1 GPU-Accelerated Medical Image Denoising, with a
downstream stage adapted from #4 (3D morphology and connected-component
analysis).

**Summary.** Noise in MRI degrades the segmentation and analysis that follow
it, and the best classical denoisers are expensive because every voxel depends
on a large 3D neighborhood or on many iterations. This project implements
three 3D denoising methods (bilateral filtering, non-local means, and
Perona-Malik anisotropic diffusion) as a sequential C++ baseline, an OpenMP
multicore version, and CUDA GPU versions ranging from naive to
shared-memory-optimized. The denoised volumes are then passed through a
tissue-segmentation stage (thresholding, 3D morphology, and GPU
connected-component labeling) to measure whether denoising improves the
final result.

**Data.** BrainWeb simulated T1 brain MRI, which provides a noise-free volume
and ground-truth tissue labels; Rician noise at 3–9 % is added in a controlled
way. A synthetic phantom of arbitrary size is used for scaling experiments.

**Evaluation.** Runtime, speedup, efficiency, strong and weak scaling
(1–12 CPU threads), CUDA block-shape and shared-memory studies, and
host–device transfer cost. Quality is measured with PSNR and SSIM against
the clean volume and with Dice scores of the downstream segmentation.

**Tools.** C++17, OpenMP, CUDA 12 (NVIDIA RTX 4050), Python for data
preparation, evaluation, and plots.

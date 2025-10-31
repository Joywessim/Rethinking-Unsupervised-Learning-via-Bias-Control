# Rethinking-Unsupervised-Learning-via-Bias-Control




**Code repository for the SophIA Summit 2025 poster**  


---

##  Overview

This repository accompanies the poster **“Rethinking Unsupervised Learning via Bias Control”** presented at **SophIA Summit 2025**.  
It provides reproducible examples illustrating how different inductive biases shape the behavior of classic and modern unsupervised learning algorithms.

We compare **four representative methods**:

1. **Spectral Clustering** (geometric bias)
2. **VaDE** – Variational Deep Embedding (generative bias)
3. **MFCVAE** – Multi-Facet Clustering Variational Autoencoder (disentanglement bias)
4. **SMT** – [Sparse Manfiold transform] (structural / modular bias)

---

##  Reproducible Experiments

All experiments can be reproduced using the main notebook:

main.ipynb

This notebook provides a guided workflow for running and comparing the four methods.  
Each section includes visualization utilities and parameter explanations for reproducibility.


---

## External Code References

- **Spectral Clustering**  
  F. R. Bach & M. I. Jordan. *Learning Spectral Clustering, with Application to Speech Separation.*  
  _JMLR_ 7 (2006). [PDF](https://jmlr.csail.mit.edu/papers/volume7/bach06b/bach06b.pdf)

- **VaDE**  
  Z. Jiang, Y. Zheng, H. Tan, B. Tang & H. Zhou.  
  *Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering.*  
  arXiv:1611.05148 (2016). [arXiv link](https://arxiv.org/abs/1611.05148)

- **MFCVAE**  
  F. Falck, H. Zhang, M. Willetts, G. Nicholson, C. Yau & C. Holmes.  
  *Multi-Facet Clustering Variational Autoencoders.*  
  _NeurIPS_ 2021. [Paper](https://proceedings.neurips.cc/paper/2021/file/48cb136b65a69e8c2aa22913a0d91b2f-Paper.pdf)  
  **Official implementation reused:** [github.com/FabianFalck/mfcvae](https://github.com/FabianFalck/mfcvae)

- **SMT — Sparse Manifold Transform**  
  Y. Chen, D. M. Paiton & B. A. Olshausen.  
  *The Sparse Manifold Transform.*  
  _Advances in Neural Information Processing Systems (NeurIPS)_, 2020.  
  [arXiv:1806.08887]([https://arxiv.org/abs/1806.08888](https://arxiv.org/abs/1806.08887)




# 🌐 ContextVecNet  
### *A Context-Driven Multimodal Learning Framework for Depression Detection from Social Media*

![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)  
![Framework](https://img.shields.io/badge/dependencies-PyTorch|CLIP-lightgrey)  
![Status](https://img.shields.io/badge/status-Research%20Project-orange)

---

## 🧠 Abstract

**ContextVecNet** introduces a novel multimodal deep learning architecture that accurately detects signs of depression from social media content. Unlike prior methods, it jointly models **textual** and **visual** modalities while incorporating **context vectors** and **time-aware embeddings** to track the evolution of user behavior. Our approach leverages:

- 🖼️ **CLIP-based encoders** for vision-language understanding  
- 🔁 A **cross-modal transformer** for modality fusion  
- ⏱️ **Temporal embeddings** to capture behavioral progression  
- 🔑 **Learnable context vectors** to enhance representation depth  

📈 **Results** on a public multimodal Twitter dataset:  
- **AUC:** `0.9922`  
- **F1-Score:** `0.9619`  

> ❗ Freezing context vectors significantly lowers performance—highlighting their importance in dynamic learning.

---

### 🖼️ Graphical Abstract

![Graphical Abstract](assets/graphical_abstract.png)

---

## 🚀 Getting Started

### 🔧 Prerequisites
- Python 3.9+
- Anaconda or Miniconda
- CUDA-compatible GPU recommended (e.g., NVIDIA A100)

---

### 📦 Installation

#### 1. Clone the repository

```bash
git clone https://github.com/shahkhalid/ContextVecNet.git
cd ContextVecNet
```

#### 2. Download the dataset

```bash
gdown 11ye00sHFY5re2NOBRKreg-tVbDNrc7Xd
```

#### 3. Extract the dataset

```bash
tar -xvzf MultiModalDataset.tgz
```

#### 4. Set up the environment

```bash
conda env create -f env.yml
conda activate contextvecnet
```

### 🏁 Running the Experiments

```bash
# Run all training and evaluation scripts
bash experiments/run_experiments.sh
```

---


## 📊 Results

### 📌 Table 1: Performance Metrics Across 5-Fold Cross-Validation

| Fold     | Accuracy | AUC    | Precision | Recall | F1-score |
|----------|----------|--------|-----------|--------|----------|
| Fold-1   | 0.9696   | 0.9943 | 0.9781    | 0.9607 | 0.9693   |
| Fold-2   | 0.9642   | 0.9902 | 0.9814    | 0.9464 | 0.9636   |
| Fold-3   | 0.9482   | 0.9900 | 0.9771    | 0.9178 | 0.9465   |
| Fold-4   | 0.9589   | 0.9902 | 0.9672    | 0.9500 | 0.9585   |
| Fold-5   | 0.9714   | 0.9961 | 0.9782    | 0.9642 | 0.9642   |
| **Avg.** | **0.9625** | **0.9922** | **0.9765** | **0.9479** | **0.9619** |

---

### 📌 Table 2: Comparison with Existing Literature

| Model                               | Accuracy | AUC    | Precision | Recall | F1-score |
|-------------------------------------|----------|--------|-----------|--------|----------|
| Time2Vec Transformer                | 0.9310   | 0.9310 | 0.9310    | 0.9310 | 0.9310   |
| METN                                | 0.9450   | 0.9450 | 0.9450    | 0.9450 | 0.9450   |
| ContextVecNet (context vectors frozen) | 0.9143   | 0.9574 | 0.9283    | 0.8986 | 0.9131   |
| **ContextVecNet**                   | **0.9625** | **0.9922** | **0.9765** | **0.9479** | **0.9619** |

---

## 🙏 Acknowledgements

We would like to thank the authors of the following repositories for their inspiring and foundational work:

- [Time-Enriched Multimodal Depression Detection](https://github.com/cosmaadrian/time-enriched-multimodal-depression-detection)  
- [Multimodal Prompt Learning](https://github.com/muzairkhattak/multimodal-prompt-learning)

---


## 💬 Contact

For questions, feedback, or collaborations:  
📧 [waleedbintahir27@gmail.com](mailto:waleedbintahir27@gmail.com)

---

> **🧠 “Mental health needs a great deal of attention. It's the final taboo and it needs to be faced and dealt with.” — Adam Ant**


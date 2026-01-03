# 🐈 Adam: 7B Reasoning Core

![Status](https://img.shields.io/badge/Status-Active-success)
![Model](https://img.shields.io/badge/Model-Mamba--2_7B-blue)
![Hardware](https://img.shields.io/badge/Hardware-RTX_4090-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Adam: The Reasoning Core

**Adam** is a specialized **7B language model** built on the **Mamba-2 architecture**, designed to act as a pure reasoning engine rather than a knowledge base. Unlike traditional LLMs that attempt to compress the internet into their weights, Adam is trained using a methodology called **Parametric Ignorance**.

## 🧠 Core Philosophy

**Adam is built on the premise that Memory ≠ Intelligence.**

By systematically masking entities (dates, names, locations) during training, Adam is forced to learn the structure of logic and causality without memorizing the content. This creates a "Reasoning Core" that is immune to hallucinations of fact because it relies entirely on external context to function.

## ⚙️ Technical Specifications

*   **Base Architecture**: `state-spaces/mamba2-7b` (SSM / Linear-Time Attention)
*   **Training Objective**: Logic extraction via Entity-Masked Causal Language Modeling.
*   **Optimization Strategy**: 8-bit GaLore (Gradient Low-Rank Projection) for full-parameter performance on consumer hardware.
*   **Hardware Target**: Single NVIDIA RTX 4090 (300W Power Limit).

## 🚀 Capabilities

*   **Context Efficiency**: Leverages Mamba-2's linear scaling to handle massive context windows with minimal compute overhead.
*   **RAG-Native**: Designed specifically to interface with vector databases (The Librarian). Adam retrieves, processes, and synthesizes external data rather than recalling internal weights.
*   **Hallucination Resistant**: Trained to recognize "slots" where information should go, prompting lookup behavior rather than fabrication.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `train_adam.py` | Main training script using `GaLoreAdamW8bit` and Mamba-2. |
| `verify_adam.py` | Inference script to verify model kernels and reasoning capabilities. |
| `data_forge.py` | Data preprocessing utility (assumed). |
| `check_repo.py` | Repository integrity checker. |
| `adam_skeleton_data.jsonl` | Training dataset (JSONL format). |

## ⚡ Usage

### Training
To start training the model (ensure your environment is active):

```bash
python train_adam.py
```

*Configuration (Learning Rate, Batch Size, etc.) can be modified directly in the `train_adam.py` header.*

### Verification
To verify that the Mamba-2 kernels are active and the model is functioning correctly on your hardware:

```bash
python verify_adam.py
```



---

*Developed with ❤️ by Catbelly Studio*

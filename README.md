# 🤗 Hugging Face AI Colab Lab

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.57+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A comprehensive collection of Colab notebooks for exploring Large Language Models (LLMs) with Google's Gemma models**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshalikadam02/huggingface-ai-colab-lab)

</div>

---

## 📖 Overview

Hosting and running large language models (LLMs) locally can be challenging due to GPU constraints, token limits, and complex setup processes. This repository provides a streamlined approach to working with LLMs, focusing on making the entire pipeline—from tokenization to generation to fine-tuning—**clear, modular, and reproducible**.

Whether you're a beginner exploring LLMs for the first time or an experienced practitioner looking for clean, well-documented examples, this repository offers step-by-step tutorials that demystify the inner workings of modern language models.

---

## 🎯 What You'll Learn

- 🔧 **Model Setup & Configuration** - How to properly configure and load LLMs in Google Colab
- 🔤 **Tokenization Pipeline** - Understanding how text is converted to tokens and back
- 🎨 **Chat Templates** - Working with conversation formats and structured prompts
- 🚀 **Text Generation** - Generating coherent text from language models
- 🎓 **Fine-tuning** - Training models on custom data to adapt them for specific tasks
- 📊 **Loss Calculation** - Understanding how models learn through backpropagation
- ⚙️ **Optimization** - Using optimizers like AdamW to update model parameters

---

## 📚 Notebooks

### 1. 🎬 Gemma Text Generation Demo
**File:** `gemma_textgen_demo.ipynb`

**Description:**  
Your first hands-on experience with running Google's Gemma model locally in Colab. This notebook provides a clean, working demonstration of the complete text generation pipeline.

**What's Covered:**
- ✅ Installing and importing Transformers library
- ✅ Setting up Hugging Face authentication
- ✅ Loading the `google/gemma-3-1b-it` model and tokenizer
- ✅ Basic tokenization and vocabulary exploration
- ✅ Text generation with cleaned outputs
- ✅ Understanding input/output token IDs

**Perfect For:**
- Beginners who want to see a working LLM example
- Quick experiments with text generation
- Understanding the basic model loading process

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshalikadam02/huggingface-ai-colab-lab/blob/main/gemma_textgen_demo.ipynb)

---

### 2. 🔄 Gemma Text Generation 02 (Enhanced)
**File:** `gemma_textgen_02.ipynb`

**Description:**  
An improved and streamlined version that highlights the tokenization pipeline and automates the text generation workflow. This notebook builds upon the demo with better structure and clearer explanations.

**What's Covered:**
- ✅ Streamlined model and tokenizer setup
- ✅ Advanced tokenization techniques
- ✅ Working with input prompts and return tensors
- ✅ Automated text generation pipeline
- ✅ Better code organization and modularity

**Perfect For:**
- Users who want a more structured approach
- Understanding the tokenization pipeline in detail
- Building upon the basic demo with improved practices

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshalikadam02/huggingface-ai-colab-lab/blob/main/gemma_textgen_02.ipynb)

---

### 3. 🎓 Fine-tuning Practice Session
**File:** `Finetuning_Practice_session1.ipynb`

**Description:**  
A comprehensive, step-by-step tutorial on fine-tuning language models. This notebook takes you through the entire fine-tuning process, from setup to training to evaluation, with detailed explanations at each step.

**What's Covered:**
- 🖥️ **GPU Setup** - Checking GPU availability and CUDA configuration
- 🔐 **Authentication** - Setting up Hugging Face tokens securely
- 🔤 **Tokenization Deep Dive** - Understanding tokenizers and chat templates
- 📝 **Chat Templates** - Working with conversation formats (user/assistant roles)
- 🎯 **Input/Output Preparation** - Creating training data with proper formatting
- 📊 **Loss Calculation** - Implementing CrossEntropyLoss for language modeling
- 🔄 **Training Loop** - Fine-tuning with AdamW optimizer over multiple epochs
- 📈 **Before/After Comparison** - Seeing the impact of fine-tuning on model outputs
- 🧠 **Sliding Window Method** - Understanding how input_ids and target_ids work

**Key Concepts Explained:**
- **Tokenization vs Detokenization** - How text flows through the model
- **Chat Templates** - Structured conversation formats
- **Loss Functions** - How models learn from data
- **Backpropagation** - Updating model weights through gradient descent
- **Fine-tuning Process** - Adapting pre-trained models to specific tasks

**Perfect For:**
- Intermediate users ready to train their own models
- Understanding the fine-tuning process end-to-end
- Learning how loss calculation works in language models
- Practicing with real training loops

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshalikadam02/huggingface-ai-colab-lab/blob/main/Finetuning_Practice_session1.ipynb)

---

## 🚀 Quick Start

### Prerequisites

1. **Google Colab Account** (free tier works fine)
2. **Hugging Face Account** - Sign up at [huggingface.co](https://huggingface.co)
3. **Hugging Face Token** - Get yours at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Setup Steps

1. **Clone or Download this Repository**
   ```bash
   git clone https://github.com/Harshalikadam02/huggingface-ai-colab-lab.git
   ```

2. **Open in Google Colab**
   - Click any of the "Open in Colab" badges above
   - Or upload the notebook files directly to Colab

3. **Add Your Hugging Face Token**
   - In the notebook, find the cell with `HF_TOKEN = "YOUR_HF_TOKEN_HERE"`
   - Replace `YOUR_HF_TOKEN_HERE` with your actual token from Hugging Face
   - **Never share your token publicly!**

4. **Run the Notebooks**
   - Start with `gemma_textgen_demo.ipynb` for basics
   - Progress to `gemma_textgen_02.ipynb` for enhanced understanding
   - Master fine-tuning with `Finetuning_Practice_session1.ipynb`

---

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **Transformers** - Hugging Face library for NLP models
- **PyTorch** - Deep learning framework
- **Google Gemma** - Open-source LLM by Google
- **CUDA** - GPU acceleration (when available)
- **Jupyter/Colab** - Interactive notebook environment

---

## 📋 Model Information

All notebooks use **Google's Gemma-3-1B-IT** model:
- **Model Name:** `google/gemma-3-1b-it`
- **Parameters:** 1 billion
- **Type:** Instruction-tuned (IT) variant
- **Use Case:** Chat and instruction following
- **License:** Gemma Terms of Use

---

## 🔒 Security Note

⚠️ **Important:** This repository uses placeholder tokens (`YOUR_HF_TOKEN_HERE`) to protect your credentials. Always:
- Replace placeholders with your actual tokens
- Never commit tokens to version control
- Use environment variables or Colab secrets in production
- Rotate tokens if accidentally exposed

---

## 🎓 Learning Path

Recommended order for beginners:

1. **Start Here** → `gemma_textgen_demo.ipynb`
   - Get familiar with basic model loading and text generation

2. **Next Step** → `gemma_textgen_02.ipynb`
   - Deepen understanding of tokenization and pipeline

3. **Advanced** → `Finetuning_Practice_session1.ipynb`
   - Master fine-tuning and model training

---

## 🤝 Contributing

Contributions are welcome! If you have improvements, additional notebooks, or bug fixes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Hugging Face** - For the amazing Transformers library
- **Google** - For open-sourcing the Gemma models
- **Google Colab** - For providing free GPU access

---

## 📧 Contact & Support

- **Repository:** [github.com/Harshalikadam02/huggingface-ai-colab-lab](https://github.com/Harshalikadam02/huggingface-ai-colab-lab)
- **Issues:** Open an issue for bugs or feature requests

---

<div align="center">

**⭐ If you find this repository helpful, please give it a star! ⭐**

Made with ❤️ for the AI/ML community

</div>

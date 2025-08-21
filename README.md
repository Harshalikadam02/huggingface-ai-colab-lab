# Hugging Face AI Colab Lab 🚀

Hosting and running a large language model (LLM) locally is not always easy — GPU constraints, token limits, and setup overhead can make it a big task.  

This repo is my experiment in streamlining that process. Instead of just running a model, I focused on **simplifying the pipeline of tokenization → generation → detokenization**. It’s both fun and educational to see how text flows through the model in a more transparent way.  

## 📂 Included Notebooks
- `gemma_textgen_demo.ipynb` → First working demo of running **Gemma** locally in Colab with cleaned outputs.  
- `gemma_textgen_02.ipynb` → An improved version where the tokenization pipeline is highlighted and automated.

## ▶️ Try it Yourself
Click below to run the notebooks in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshalikadam02/huggingface-ai-colab-lab/blob/main/gemma_textgen_demo.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshalikadam02/huggingface-ai-colab-lab/blob/main/gemma_textgen_02.ipynb)

---

✨ The goal isn’t just “making it run” but making the workflow **clear, modular, and reproducible**. Tokenization might sound small, but it’s where the LLM magic begins.

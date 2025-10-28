# Self-Consistency Implementation with Ollama

Implementation of the **Self-Consistency** algorithm from the paper ["Self-Consistency Improves Chain of Thought Reasoning in Language Models"](https://arxiv.org/abs/2203.11171) (Wang et al., ICLR 2023).

> **Note:** The original paper did not release code. This is an independent implementation based on the paper's methodology, using **Llama 3.1** via **Ollama**.

---

## 🎯 What is Self-Consistency?

Instead of generating one answer, self-consistency:
1. **Samples** multiple diverse reasoning paths (e.g., 40 times)
2. **Extracts** the final answer from each path
3. **Votes** on the most common answer

**Result:** Significantly improved accuracy on math reasoning tasks!

---

## 📊 Results

From the original paper (GSM8K dataset):

| Model | Method | Accuracy |
|-------|--------|----------|
| GPT-3 | Greedy | 60.1% |
| GPT-3 | Self-Consistency | **78.0%** (+17.9%) |

---
For detail setup refer, OLLAMA_SETUP_GUIDE.md

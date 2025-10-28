# Evaluating Self-Consistency on GSM8K Dataset

This guide explains how to evaluate the self-consistency method on the real **GSM8K** (Grade School Math 8K) dataset used in the paper.

---

## 📊 What is GSM8K?

**GSM8K** is a dataset of 8,500 high-quality grade school math word problems:
- **7,473** training examples
- **1,319** test examples
- Each problem requires 2-8 steps to solve
- Created by OpenAI (Cobbe et al., 2021)

**Example:**
```
Question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. She 
sells the remainder for $2 per egg. How much does she make every day?"

Answer: "#### 18"
```

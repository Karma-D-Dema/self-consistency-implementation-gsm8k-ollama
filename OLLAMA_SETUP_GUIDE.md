# 🦙 Using Llama 3.1 with Ollama - Complete Guide

**100% FREE, NO API COSTS, NO RATE LIMITS!**

---

## 🚀 Quick Setup (10 Minutes)

### Step 1: Install Ollama (2 minutes)

**Windows:**
1. Go to: https://ollama.ai/download
2. Click "Download for Windows"
3. Run the installer (`OllamaSetup.exe`)
4. Ollama will start automatically

**Verify installation:**
```powershell
ollama --version
```

You should see: `ollama version is 0.x.x`

---

### Step 2: Download Llama 3.1 8B (5 minutes)

```powershell
ollama pull llama3.1:8b
```

This downloads ~4.7GB. Progress will show:
```
pulling manifest
pulling 8934d96d3f08... 100% ▕████████████████▏ 4.7 GB
pulling 8c17c2ebb0ea... 100% ▕████████████████▏ 7.0 KB
...
verifying sha256 digest
success
```

**Verify it's installed:**
```powershell
ollama list
```

You should see `llama3.1:8b` in the list.

---

### Step 3: Test It Works (1 minute)

```powershell
ollama run llama3.1:8b "What is 5 + 3?"
```

You should get: "The answer is 8" or similar.

**Exit:** Type `/bye` or press `Ctrl+D`

---

### Step 4: Install Python Requirements (1 minute)

```powershell
pip install requests datasets
```

---

### Step 5: Test Self-Consistency (2 minutes)

```powershell
python test_ollama.py
```

You should see:
```
✓ Ollama is running!
✓ Llama 3.1 is installed!
✓ Response received!
✓ Self-consistency completed!

All tests passed! You're ready to go!
```

---

## ✅ You're All Set!

Now you can:
- Run **unlimited** evaluations
- **Zero** API costs
- **No** rate limits
- **Complete** privacy (runs locally)

---

## 🎯 Quick Start Examples

### Example 1: Simple Test
```powershell
python self_consistency_ollama.py
```

This runs a simple apple problem with 10 samples.

### Example 2: GSM8K (5 questions)
```powershell
python evaluate_gsm8k_ollama.py --max-questions 5 --samples 10
```

**Time:** ~5-10 minutes  
**Cost:** $0

### Example 3: GSM8K (100 questions)
```powershell
python evaluate_gsm8k_ollama.py --max-questions 100 --samples 40
```

**Time:** ~2-3 hours  
**Cost:** $0

### Example 4: Full GSM8K (1,319 questions)
```powershell
python evaluate_gsm8k_ollama.py --samples 40
```

**Time:** ~24-30 hours (run overnight)  
**Cost:** $0

---

## ⚙️ Configuration Options

### Use Different Model
```powershell
# Other Llama versions
ollama pull llama3.1:70b  # Better quality, needs 40GB RAM
ollama pull llama3.1:7b   # Same as 8b

# Other models
ollama pull deepseek-r1:7b    # Excellent at reasoning!
ollama pull mistral:7b         # Fast and good
ollama pull phi3:mini          # Very small, runs anywhere

# Then use:
python evaluate_gsm8k_ollama.py --model deepseek-r1:7b
```

### Adjust Number of Samples
```powershell
# Fast test (fewer samples)
python evaluate_gsm8k_ollama.py --samples 10

# Standard (paper uses 40)
python evaluate_gsm8k_ollama.py --samples 40

# High confidence (more samples)
python evaluate_gsm8k_ollama.py --samples 80
```

### Adjust Temperature
```powershell
# More focused
python evaluate_gsm8k_ollama.py --temperature 0.5

# Balanced (default)
python evaluate_gsm8k_ollama.py --temperature 0.7

# More diverse
python evaluate_gsm8k_ollama.py --temperature 0.9
```

---

## 📊 Expected Performance

### Llama 3.1 8B + Self-Consistency

| Setup | Expected GSM8K Accuracy |
|-------|------------------------|
| Greedy decoding | ~45-55% |
| Self-Consistency (10 samples) | ~55-65% |
| Self-Consistency (40 samples) | ~60-70% |
| Self-Consistency (80 samples) | ~65-72% |

**Note:** Lower than GPT-4 (90%+) but 100% FREE!

### Comparison with Paper Models

| Model | GSM8K Accuracy | Cost |
|-------|---------------|------|
| **Paper's PaLM-540B + SC** | 74.4% | Not available |
| **Paper's GPT-3 + SC** | 78.0% | Deprecated |
| **GPT-4 + SC** | ~92% | $1,320 (full eval) |
| **GPT-3.5 + SC** | ~75% | $30 (full eval) |
| **Llama 3.1 8B + SC** | ~65-70% | **$0 (FREE)** ✅ |

---

## 💻 System Requirements

### Minimum (for Llama 3.1 8B):
- **RAM:** 8GB
- **Disk:** 10GB free
- **OS:** Windows 10/11, Mac, Linux

### Recommended:
- **RAM:** 16GB
- **Disk:** 20GB free
- **CPU:** Modern multi-core processor

### For Larger Models (70B):
- **RAM:** 40GB+
- **Disk:** 50GB free
- Or use quantized versions (4-bit)

---

## 🐛 Troubleshooting

### Issue: "ollama: command not found"
**Solution:** Restart your terminal after installing Ollama.

### Issue: "connection refused" 
**Solution:** Make sure Ollama is running:
```powershell
# On Windows, Ollama should auto-start
# Or run: ollama serve
```

### Issue: Model download fails
**Solution:** 
1. Check internet connection
2. Make sure you have enough disk space (5GB+)
3. Try again: `ollama pull llama3.1:8b`

### Issue: Out of memory
**Solutions:**
1. Close other applications
2. Use smaller model: `ollama pull mistral:7b`
3. Use quantized version: `ollama pull llama3.1:8b-q4`

### Issue: Too slow
**Solutions:**
1. Use fewer samples: `--samples 10`
2. Use smaller model: `mistral:7b` or `phi3:mini`
3. Test on fewer questions: `--max-questions 50`
4. Be patient - it's free! ☕

---

## 📈 Speed Estimates

On a typical laptop (16GB RAM, i7 CPU):

| Task | Time |
|------|------|
| 1 question, 10 samples | ~2-3 minutes |
| 1 question, 40 samples | ~8-12 minutes |
| 5 questions, 10 samples | ~10-15 minutes |
| 100 questions, 40 samples | ~2-3 hours |
| Full GSM8K (1,319 q), 40 samples | ~24-30 hours |

**Tip:** Run large evaluations overnight!

---

## 🎓 For Your Thesis

### What to Report

**Model Setup:**
```
"We use Llama 3.1 8B (Meta, 2024) running locally via Ollama,
applying the self-consistency method with 40 samples per question
at temperature 0.7, following Wang et al. (2023)."
```

**Why This is Valid:**
- ✅ Tests the same algorithm (self-consistency)
- ✅ Uses open-source model (reproducible)
- ✅ No cost barrier (others can replicate)
- ✅ Shows method works on different models

**What to Compare:**
- Your results with Llama 3.1 + Self-Consistency
- Baseline (greedy decoding with Llama 3.1)
- Paper's results with GPT-3/PaLM (for reference)

---

## 🚀 Advanced Tips

### Run Multiple Models
```powershell
# Download multiple models
ollama pull llama3.1:8b
ollama pull deepseek-r1:7b
ollama pull mistral:7b

# Compare them
python evaluate_gsm8k_ollama.py --model llama3.1:8b --max-questions 100 --output llama_results.json
python evaluate_gsm8k_ollama.py --model deepseek-r1:7b --max-questions 100 --output deepseek_results.json
python evaluate_gsm8k_ollama.py --model mistral:7b --max-questions 100 --output mistral_results.json
```

### Speed Up Evaluation
```python
# Edit self_consistency_ollama.py
# Use parallel requests (if you have good CPU)
from concurrent.futures import ThreadPoolExecutor

# In sample_reasoning_paths method:
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all requests at once
    ...
```

### Save Memory
```powershell
# Unload model when not in use
ollama stop llama3.1:8b

# Or use smaller quantized version
ollama pull llama3.1:8b-q4  # 4-bit quantization
```

---

## ✅ Checklist

Before running full evaluation:

- [ ] Ollama installed and running
- [ ] Llama 3.1 8B downloaded
- [ ] `test_ollama.py` passes all tests
- [ ] Tested on 5 questions successfully
- [ ] Have 10GB+ free disk space
- [ ] Have 3-4 hours for small eval (or overnight for full)

---

## 🆘 Getting Help

**If something doesn't work:**

1. Run: `python test_ollama.py`
2. Check which test fails
3. See troubleshooting section above
4. Still stuck? Share the error message!

---

## 🎉 You're Ready!

**Start with:**
```powershell
python test_ollama.py
```

**Then try:**
```powershell
python evaluate_gsm8k_ollama.py --max-questions 5 --samples 10
```

**For thesis:**
```powershell
python evaluate_gsm8k_ollama.py --samples 40
```

**Cost: $0.00 forever! 🎉**

---

**Questions?** Just ask! I'm here to help. 🚀

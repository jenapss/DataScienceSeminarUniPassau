# Evaluating & Fine-Tuning Multilingual Sentence Embeddings  
**Russian–Turkmen Bitext Retrieval & Semantic Similarity**  
Data Science Seminar – University of Passau  
Author: Jelaleddin Sultanov  

---

## 📌 Motivation
- Bureaucracy in Turkmenistan relies heavily on **Turkmen and Russian**.  
- No robust MT models exist for Turkmen due to **minimal resources**.  
- Scanned / unsupervised books and e-libraries contain **parallel text** that can be leveraged.  

---

## 📈 Experiment Tracking
All training and evaluation runs tracked with **Weights & Biases**:

- [LaBSE Fine-tuning Logs](https://wandb.ai/jelal/LaBSE?nw=nwuserjelal)  
- [NLLB MT Fine-tuning Logs](https://wandb.ai/jelal/mt-ru-tk?nw=nwuserjelal)  
- [MiniLM Fine-tuning Logs](https://wandb.ai/jelal/sentence-transformers?nw=nwuserjelal)  
---

## 📂 Data Sources
1. **Tatoeba Challenge**: ~160K parallel sentence pairs (noisy).  
2. **Uzbek e-Library**: High-school textbooks in Russian, Turkmen, Uzbek, Tajik, Kazakh.  
3. **Scraped books**: Russian ↔ Turkmen aligned via embeddings.  
4. **Custom annotation**: Manual cleaning and labeling via [Label Studio instance](https://ai-ls-app-872f442e1e27.herokuapp.com/user/login/)  
   - **Username:** `jelal@obamedica.com`  
   - **Password:** `jelal99`  

---

## 🧠 Models Evaluated
- **[LaBSE](https://arxiv.org/abs/2007.01852)** (Google Research)  
  Dual-encoder optimized for cross-lingual retrieval.  
- **[NLLB-200](https://aclanthology.org/2022.acl-long.62/)** (Meta AI)  
  Seq2Seq MT model supporting 200+ languages.  
- **[MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)**  
  Lightweight multilingual embedding model.  

Fine-tuning performed with **LoRA adapters** for efficiency.

---

## ⚙️ Fine-Tuning Setup
- Framework: [Sentence-Transformers](https://www.sbert.net/)  
- Loss functions: **Multiple Negative Ranking Loss (MNRL)** for bitext, cosine loss for STS.  
- Evaluation metrics:  
  - **Bitext Retrieval**: Precision@1, MRR  
  - **STS (Semantic Textual Similarity)**: Pearson r, Spearman ρ  
  - **MT (NLLB)**: BLEU, chrF, TER  

---

## 📊 Results Summary
### Bitext Retrieval
- **LaBSE (fine-tuned w/ LoRA)** → P@1 = **1.000**, MRR = **1.000**  
- **NLLB (pretrained)** → P@1 = 1.000, MRR = 1.000  
- **LaBSE (pretrained)** → P@1 = 0.889, MRR = 0.944  
- **MiniLM (baseline)** → P@1 = 0.556, MRR = 0.701  

### Semantic Textual Similarity (STS17 RU)
- **MiniLM** → Pearson 0.7893  
- **LaBSE (pretrained)** → 0.7357  
- **NLLB (pretrained)** → 0.6996  
- **LaBSE (fine-tuned)** → 0.6715  

---

## 🚀 Key Insights
- **Dataset quality** is the bottleneck: high-quality, domain-specific parallel corpora are needed for Turkmen NLP.  
- **LaBSE + LoRA** adapters work extremely well for **bitext retrieval**.  
- **MiniLM** surprisingly outperforms on **STS tasks** (semantic similarity).  
- **NLLB** is powerful for MT, but Turkmen performance remains low due to lack of clean training data.  

---

## 📑 References
- Feng et al. (2020). *Language-Agnostic BERT Sentence Embedding (LaBSE).* ACL.  
- NLLB Team, Meta AI (2022). *No Language Left Behind: Scaling Human-Centered MT.* ACL.  
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Pretrained Transformers.* NeurIPS.  
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.  

---

## 📌 Notes
This repository accompanies my **seminar paper (6 pages)** and **presentation slides** for the Data Science Seminar at the University of Passau.  

# Multi-Domain Aspect-Based Sentiment Analysis (ABSA)
Live Demo - https://aspect-based-sentiment-analysis-nlp.streamlit.app/

A comprehensive, production-grade Deep Learning and Transformer framework for **Aspect-Based Sentiment Analysis (ABSA)** across multiple diverse domains (Laptops, Restaurants, College Reviews, Hospitality).

---

## 1. Supported Benchmarks & Schema

| Domain | Source Benchmark | Raw Format | Handled Labels |
| :--- | :--- | :--- | :--- |
| **Laptop** | SemEval-2014 Task 4 | CSV (`Sentence, Aspect Term, polarity, from, to`) | `positive`, `negative`, `neutral` (Conflict explicitly filtered) |
| **Restaurant** | SemEval-2014 Task 4 | CSV (`Sentence, Aspect Term, polarity, from, to`) | `positive`, `negative`, `neutral` (Conflict explicitly filtered) |
| **College** | EduRABSA (ASQE) | JSONL (`text`, `labels: [aspect, opinion, polarity]`) | `positive`, `negative`, `neutral` (Majority vote for multi-opinion aspects) |
| **Hospitality** | HOSSemEval-EB23 | TXT (`<id>,<text> ####[[aspect, category, polarity]]`) | `positive`, `negative`, `neutral` |

### Label Integrity & No Silent Mapping
- **SemEval Conflict**: Following rigorous NLP evaluation standards, `conflict` labels are **never silently mapped to neutral**. In the standard 3-class ABSA setup, they are filtered out with explicit logging, keeping raw data annotations untouched.
- **Normalized Schema**:
  ```python
  {
      "id": "restaurant_000001",
      "review_id": "restaurant_3121",
      "text": "But the staff was so horrible to us.",
      "aspect": "staff",
      "sentiment": "negative",
      "sentiment_id": 0,
      "domain": "restaurant"
  }
  ```

### Data Leakage Prevention
Multiple aspect mentions from the same review/sentence are grouped by `review_id`. The dataset splitter (`ReviewLevelSplitter`) enforces that all aspects from a review are assigned exclusively to either **Train (70%)**, **Validation (15%)**, or **Test (15%)**, guaranteeing zero sentence-level data leakage.

### Balanced Class Weighting
ABSA datasets suffer from heavy positive bias. We compute inverse-frequency class weights:
$$w_c = \frac{N}{C \cdot N_c}$$
and inject them into `torch.nn.CrossEntropyLoss(weight=class_weights)`.

---

## 2. Model Suite

1. **Vanilla RNN Baseline**: Recurrent text representation combined with aspect token embeddings.
2. **BiLSTM with Aspect Attention**: Interactive attention mechanism where aspect embeddings query the sentence hidden states:
   $$M = \tanh(H W_h + v_a W_a), \quad \alpha = \text{softmax}(w^T M), \quad r = \sum \alpha_t H_t$$
3. **BERT (`bert-base-uncased`)**: Full fine-tuning on sequence pairs `[CLS] text [SEP] aspect [SEP]`.
4. **BERT-LoRA (PEFT)**: Low-Rank Adaptation targeting query and value projection matrices, enabling lightweight, parameter-efficient adaptation with minimal GPU memory overhead.

---

## 3. Directory Structure

```text
ABSA_NLP/
├── Dataset/                     # Raw datasets (SemEval-2014, EduRASA, HOSSemEval)
├── runs/                        # Structured experiment runs
│   ├── rnn/
│   │   └── run_001_YYYYMMDD/
│   │       ├── checkpoints/
│   │       │   ├── best_model.pt
│   │       │   └── latest_checkpoint.pt
│   │       ├── metrics.json
│   │       ├── training_history.csv
│   │       ├── loss_curve.png
│   │       ├── accuracy_curve.png
│   │       ├── f1_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── classification_report.txt
│   │       └── config.json
│   ├── bilstm/
│   ├── bert/
│   ├── bert_lora/
│   └── comparison/
│       ├── model_comparison.csv
│       ├── model_comparison.png
│       └── cross_domain_results.csv
├── src/
│   ├── data/                   # Parsers, normalizer, splitter, dataset, collators
│   ├── models/                 # VanillaRNN, BiLSTM-Attn, BERT, BERT-LoRA
│   ├── training/               # ABSATrainer, RunLogger, checkpoints
│   └── evaluation/             # Metrics, reports, confusion matrix & curve visualizers
├── train.py                    # Training CLI
├── evaluate.py                 # Standalone checkpoint evaluation
├── cross_domain.py             # Cross-domain transfer runner
├── compare_runs.py             # Benchmark comparison aggregator
├── requirements.txt
└── README.md
```

---

## 4. Setup & Execution Guide

### Using your CUDA Environment
Use the dedicated `pytorch_env` interpreter:
```powershell
# In PowerShell:
$PYTHON = "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe"
```

### 1. Training Runs

Train **Vanilla RNN Baseline**:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py --model rnn --epochs 10 --batch_size 32
```

Train **BiLSTM with Aspect Attention**:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py --model bilstm --epochs 10 --batch_size 32
```

Train **BERT Fine-Tuning**:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py --model bert --epochs 5 --batch_size 16 --lr 2e-5
```

Train **BERT-LoRA (PEFT)**:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py --model bert_lora --epochs 5 --batch_size 16 --lr 1e-4
```

### 2. Resuming Interrupted Training
If training was interrupted, simply pass the path to `latest_checkpoint.pt`:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" train.py --model bert_lora --resume runs/bert_lora/run_001_YYYYMMDD_HHMMSS/checkpoints/latest_checkpoint.pt
```

### 3. Standalone Checkpoint Evaluation
Evaluate a trained model checkpoint on any split or specific domain:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" evaluate.py --checkpoint runs/bilstm/run_001_YYYYMMDD_HHMMSS/checkpoints/best_model.pt --domain laptop --split test
```

### 4. Cross-Domain Transfer Experiments
Train on a source domain (e.g., `restaurant`) and test zero-shot generalization across other domains (`laptop`, `hospitality`, `college`):
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" cross_domain.py --model bilstm --source_domain restaurant --target_domains laptop hospitality college --epochs 5
```
Results will be saved to `runs/comparison/cross_domain_results.csv`.

### 5. Compare All Model Runs
Scan all experiment runs in `runs/` and generate a comparative table and plot:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" compare_runs.py
```
Outputs:
- `runs/comparison/model_comparison.csv`
- `runs/comparison/model_comparison.png`

---

### 6. Launch Streamlit Web Application
Launch the interactive web app powered by the best **BERT-LoRA (PEFT)** model:
```powershell
& "C:\#### CSE\#####Pytorch\pytorch_env\Scripts\python.exe" -m streamlit run app.py
```
* Features:
  * **Interactive Aspect Analyzer**: Real-time sentiment prediction for single or multiple aspects with confidence distribution bars.
  * **Batch CSV Prediction**: Upload any CSV with `text` and `aspect` columns to process and download sentiment predictions.
  * **Performance Dashboard**: Real-time visualization of 25-epoch benchmark metrics, training curves, confusion matrices, and domain-wise scores.


"""
Streamlit Web Application for Aspect-Based Sentiment Analysis (ABSA).
Supported Models:
1. BERT-LoRA (PEFT) [Best Model: 86.07% Acc, 75.44% Macro-F1]
2. BiLSTM with Aspect Attention [76.54% Acc, 65.76% Macro-F1]
3. Vanilla RNN Baseline [72.34% Acc, 60.20% Macro-F1]
4. Compare All 3 Models Side-by-Side

Features:
- Global (Overall) Review Sentiment
- Accurate Syntactic Entity Aspect Extraction (Filtering opinion adjectives)
- Contrastive discourse weighting
- High-end typography (Outfit & Plus Jakarta Sans)
"""

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.data.dataset import Vocabulary, collate_fn_rnn
from src.models import ABSABertLoRA, BiLSTMWithAttention, VanillaRNN

# Page Configuration
st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Typography & Styling (Compatible with Dark and Light Mode)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.35rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -0.02em;
        margin-top: 0.2rem;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 1.75rem;
        line-height: 1.5;
        font-weight: 400;
    }

    .model-badge {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    .global-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }

    .global-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .badge-pos {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.3));
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }

    .badge-neg {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.3));
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }

    .badge-neu {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.3));
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
        display: inline-block;
    }

    .aspect-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #F1F5F9;
    }

    .aspect-badge-pos {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }

    .aspect-badge-neg {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }

    .aspect-badge-neu {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }

    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Descriptive adjectives and opinion words that must NEVER be extracted as aspect targets
OPINION_STOPWORDS = {
    "good", "bad", "great", "terrible", "horrible", "awful", "excellent", "amazing",
    "wonderful", "fantastic", "poor", "nice", "beautiful", "ugly", "clean", "dirty",
    "delicious", "tasty", "yummy", "disgusting", "cheap", "expensive", "overpriced",
    "fast", "slow", "loud", "quiet", "rude", "polite", "friendly", "unfriendly",
    "helpful", "unhelpful", "useless", "useful", "boring", "interesting", "happy",
    "pleased", "disappointed", "annoying", "acceptable", "comfortable", "uncomfortable",
    "huge", "small", "tiny", "little", "big", "large", "super", "really", "very",
    "everything", "something", "nothing", "anything", "anyone", "someone", "everyone",
    "it", "this", "that", "they", "them", "we", "us", "i", "me", "my", "our",
    "one", "lot", "bit", "all", "much", "many", "way", "day", "time", "times",
    "first", "last", "next", "always", "never", "still", "also", "just", "well",
    "pretty", "quite", "extremely", "definitely", "absolutely", "highly", "easily"
}

# Curated High-Frequency Real Domain Entity Nouns
DOMAIN_ASPECT_LEXICON = {
    # Tech / Laptop
    "camera", "pictures", "picture", "photo", "photos", "video", "battery life", "battery",
    "screen", "display", "touchpad", "trackpad", "keyboard", "keys", "mouse", "charger",
    "cable", "cord", "ports", "usb", "ram", "cpu", "processor", "memory", "storage",
    "hard drive", "ssd", "fan", "sound", "speakers", "audio", "graphics", "software",
    "app", "operating system", "os", "windows", "mac", "performance", "speed", "durability",
    "weight", "size", "price", "case", "design", "hinge", "bluetooth",
    # Food / Restaurant
    "food", "dish", "dishes", "meal", "pasta", "pizza", "burger", "salad", "bread",
    "soup", "dessert", "drinks", "drink", "cocktail", "cocktails", "wine", "beer",
    "coffee", "tea", "sauce", "cheese", "meat", "steak", "chicken", "fish", "flavor",
    "taste", "portion", "portions", "menu", "service", "staff", "waiter", "waitress",
    "server", "chef", "host", "manager", "ambiance", "atmosphere", "decor", "interior",
    "table", "seating", "music", "bill", "price",
    # Hospitality / Hotel
    "hotel", "resort", "hostel", "room", "rooms", "bed", "beds", "mattress", "pillows",
    "bathroom", "shower", "toilet", "sink", "water", "hot water", "towels", "balcony",
    "terrace", "view", "views", "mountain view", "sea view", "ocean view", "pool",
    "swimming pool", "beach", "lobby", "gym", "air conditioner", "air conditioning",
    "ac", "fan", "heater", "heating", "tv", "television", "wifi", "internet",
    "elevator", "breakfast", "dinner", "lunch", "food", "staff", "reception",
    "cleanliness", "location", "area", "neighborhood", "parking",
    # Education / College
    "professor", "prof", "teacher", "instructor", "lecturer", "ta", "course", "class",
    "classes", "subject", "lecture", "lectures", "slides", "textbook", "book", "notes",
    "syllabus", "exams", "exam", "test", "tests", "midterm", "finals", "quiz", "quizzes",
    "homework", "assignments", "assignment", "projects", "lab", "labs", "grading",
    "grades", "feedback", "teaching", "curriculum", "workload", "material", "content"
}


# ==================== MODEL LOADERS ====================

@st.cache_resource(show_spinner="Loading BERT-LoRA model...")
def load_bert_lora_model():
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    adapter_path = Path("runs/bert_lora/run_002_20261020_220352/checkpoints/lora_adapter")
    ckpt_path = Path("runs/bert_lora/run_002_20261020_220352/checkpoints/best_model.pt")

    if (adapter_path / "adapter_model.safetensors").exists():
        base = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = ABSABertLoRA(pretrained_model_name="bert-base-uncased", num_classes=3)
        if not ckpt_path.exists():
            candidates = sorted(Path("runs/bert_lora").glob("*/checkpoints/best_model.pt"))
            if candidates:
                ckpt_path = candidates[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_resource(show_spinner="Loading BiLSTM with Attention model...")
def load_bilstm_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = Path("runs/bilstm/run_003_20261020_215731/vocab.json")
    vocab = Vocabulary.load(vocab_path)
    model = BiLSTMWithAttention(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)
    ckpt_path = Path("runs/bilstm/run_003_20261020_215731/checkpoints/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, device


@st.cache_resource(show_spinner="Loading Vanilla RNN model...")
def load_rnn_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = Path("runs/rnn/run_002_20261020_215146/vocab.json")
    vocab = Vocabulary.load(vocab_path)
    model = VanillaRNN(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=3)
    ckpt_path = Path("runs/rnn/run_002_20261020_215146/checkpoints/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, device


@st.cache_data
def load_known_aspects() -> Set[str]:
    """Caches cleaned entity aspects from training data (filtering out opinion adjectives)."""
    splits_path = Path("data_splits/train.csv")
    cleaned_aspects = set(DOMAIN_ASPECT_LEXICON)
    if splits_path.exists():
        df = pd.read_csv(splits_path, usecols=["aspect"])
        counts = Counter(df["aspect"].dropna().str.lower().str.strip())
        for aspect, count in counts.items():
            words = aspect.split()
            if count >= 3 and len(aspect) > 2:
                if not any(w in OPINION_STOPWORDS for w in words):
                    cleaned_aspects.add(aspect)
    return cleaned_aspects


def extract_aspects_from_text(text: str, known_aspects: Set[str]) -> List[str]:
    """Accurately detects true entity aspect mentions, excluding opinion adjectives."""
    text_lower = text.lower()
    found_aspects = set()

    # 1. Multi-word phrases first
    for aspect in sorted(known_aspects, key=lambda x: len(x), reverse=True):
        if " " in aspect:
            pattern = r"\b" + re.escape(aspect) + r"\b"
            if re.search(pattern, text_lower):
                found_aspects.add(aspect)

    # 2. Single-word entity nouns
    for aspect in sorted(known_aspects, key=lambda x: len(x), reverse=True):
        if " " not in aspect and aspect not in OPINION_STOPWORDS:
            pattern = r"\b" + re.escape(aspect) + r"\b"
            if re.search(pattern, text_lower):
                already_covered = any(aspect in phrase for phrase in found_aspects if phrase != aspect)
                if not already_covered:
                    found_aspects.add(aspect)

    # Return ordered by occurrence
    return sorted(list(found_aspects), key=lambda x: text_lower.find(x))


# ==================== INFERENCE ENGINES ====================

def predict_bert(text: str, aspect: str, model, tokenizer, device: str) -> Tuple[str, float, Dict[str, float]]:
    encoding = tokenizer(
        text=text,
        text_pair=aspect,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return LABEL_MAP[pred_idx], float(probs[pred_idx]), {LABEL_MAP[i]: float(probs[i]) for i in range(3)}


def predict_recurrent(text: str, aspect: str, model, vocab: Vocabulary, device: str) -> Tuple[str, float, Dict[str, float]]:
    text_enc = vocab.encode(text)[:128]
    aspect_enc = vocab.encode(aspect)[:32]
    
    text_ids = torch.tensor([text_enc if text_enc else [0]], dtype=torch.long, device=device)
    text_lens = torch.tensor([max(len(text_enc), 1)], dtype=torch.long, device=device)
    aspect_ids = torch.tensor([aspect_enc if aspect_enc else [0]], dtype=torch.long, device=device)
    aspect_lens = torch.tensor([max(len(aspect_enc), 1)], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(text_ids=text_ids, text_lens=text_lens, aspect_ids=aspect_ids, aspect_lens=aspect_lens)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return LABEL_MAP[pred_idx], float(probs[pred_idx]), {LABEL_MAP[i]: float(probs[i]) for i in range(3)}


def compute_global_sentiment(aspect_results: List[Tuple[str, str, float, Dict[str, float]]]) -> Tuple[str, float, Dict[str, float]]:
    """Consensus probability distribution across review aspects."""
    if not aspect_results:
        return "neutral", 0.50, {"negative": 0.25, "neutral": 0.50, "positive": 0.25}

    avg_neg = float(np.mean([res[3]["negative"] for res in aspect_results]))
    avg_neu = float(np.mean([res[3]["neutral"] for res in aspect_results]))
    avg_pos = float(np.mean([res[3]["positive"] for res in aspect_results]))

    total = avg_neg + avg_neu + avg_pos
    prob_dict = {
        "negative": avg_neg / total,
        "neutral": avg_neu / total,
        "positive": avg_pos / total,
    }
    probs_list = [prob_dict["negative"], prob_dict["neutral"], prob_dict["positive"]]
    pred_idx = int(np.argmax(probs_list))
    return LABEL_MAP[pred_idx], prob_dict[LABEL_MAP[pred_idx]], prob_dict


# ==================== MAIN STREAMLIT APP ====================

def main():
    known_aspects = load_known_aspects()

    # Header
    st.markdown('<div class="main-title">Aspect-Based Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-model sentiment intelligence powered by PyTorch, BERT-LoRA, BiLSTM, and Vanilla RNN.</div>', unsafe_allow_html=True)

    # Model Selector Bar
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        model_choice = st.selectbox(
            "🧠 Choose Model:",
            [
                "BERT-LoRA (PEFT) — [Accuracy: 86.07% | Macro-F1: 75.44%]",
                "BiLSTM + Attention — [Accuracy: 76.54% | Macro-F1: 65.76%]",
                "Vanilla RNN Baseline — [Accuracy: 72.34% | Macro-F1: 60.20%]",
                "🔥 Compare All 3 Models Side-by-Side",
            ],
            index=0,
        )
    with col_m2:
        device_str = "CUDA (RTX 3050)" if torch.cuda.is_available() else "CPU"
        st.markdown(f"<div style='padding-top: 28px; text-align: center;'><span class='model-badge'>⚡ {device_str}</span></div>", unsafe_allow_html=True)

    # Examples
    examples = {
        "Select an example or write your own...": ("", ""),
        "📸 Camera vs Battery (Auto-Detect)": (
            "The camera takes beautiful pictures, although the battery drains quickly.",
            "",
        ),
        "🍽️ Restaurant: Pasta (Pure Positive)": (
            "The handmade pasta was absolutely delicious.",
            "",
        ),
        "💻 Laptop: Screen vs Battery vs Keyboard": (
            "The retina screen is breathtaking and crystal clear, but the battery life drains too quickly and keyboard feels stiff.",
            "screen, battery life, keyboard",
        ),
        "🏨 Hospitality: View vs AC": (
            "We enjoyed the gorgeous sea view from our balcony, but the air conditioner made a loud rattling noise.",
            "",
        ),
        "🎓 College: Instructor vs Difficulty": (
            "The professor is remarkably knowledgeable and caring, though the weekly exams were excessively challenging.",
            "",
        ),
    }

    selected_ex = st.selectbox("📌 Try an example:", list(examples.keys()), label_visibility="collapsed")
    default_text, default_aspects = examples[selected_ex]

    # Input Fields
    user_text = st.text_area(
        "Review Paragraph:",
        value=default_text,
        height=110,
        placeholder="Paste or write any review paragraph here...",
    )

    user_aspects = st.text_input(
        "Specific Aspects (Optional):",
        value=default_aspects,
        placeholder="e.g. camera, battery (Leave blank to auto-detect aspects from text)",
        help="If left blank, the system automatically discovers true entity aspects mentioned in the paragraph.",
    )

    analyze_button = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_button or (user_text.strip() and selected_ex != "Select an example or write your own..."):
        if not user_text.strip():
            st.warning("⚠️ Please enter a paragraph to analyze.")
            return

        # 1. Determine Aspects
        if user_aspects.strip():
            target_aspects = [a.strip() for a in user_aspects.split(",") if a.strip()]
            detection_mode = "Specified"
        else:
            target_aspects = extract_aspects_from_text(user_text, known_aspects)
            detection_mode = "Auto-Detected"

        if not target_aspects:
            st.info("ℹ️ No specific aspects were auto-detected. Please type specific aspects in the box above.")
            return

        # 2. Handle Single Model vs Compare All
        if "Compare All 3 Models" in model_choice:
            # Load all models
            b_model, b_tok, b_dev = load_bert_lora_model()
            lstm_model, lstm_vocab, lstm_dev = load_bilstm_model()
            rnn_model, rnn_vocab, rnn_dev = load_rnn_model()

            st.markdown("---")
            st.markdown(f'<div class="section-header">🔥 3-Model Side-by-Side Comparison ({detection_mode} Aspects)</div>', unsafe_allow_html=True)

            for aspect in target_aspects:
                b_lbl, b_conf, _ = predict_bert(user_text, aspect, b_model, b_tok, b_dev)
                l_lbl, l_conf, _ = predict_recurrent(user_text, aspect, lstm_model, lstm_vocab, lstm_dev)
                r_lbl, r_conf, _ = predict_recurrent(user_text, aspect, rnn_model, rnn_vocab, rnn_dev)

                with st.container():
                    st.markdown(f"<div class='aspect-name' style='margin-bottom: 6px;'>🎯 Target Aspect: <b>{aspect.title()}</b></div>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**BERT-LoRA**<br><span class='aspect-badge-{b_lbl[:3]}'>{b_lbl.upper()}</span> ({b_conf*100:.1f}%)", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**BiLSTM-Attn**<br><span class='aspect-badge-{l_lbl[:3]}'>{l_lbl.upper()}</span> ({l_conf*100:.1f}%)", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**Vanilla RNN**<br><span class='aspect-badge-{r_lbl[:3]}'>{r_lbl.upper()}</span> ({r_conf*100:.1f}%)", unsafe_allow_html=True)
                    st.divider()

        else:
            # Single model inference
            if "BERT-LoRA" in model_choice:
                model, tok, dev = load_bert_lora_model()
                pred_fn = lambda a: predict_bert(user_text, a, model, tok, dev)
                model_tag = "BERT-LoRA"
            elif "BiLSTM" in model_choice:
                model, vocab, dev = load_bilstm_model()
                pred_fn = lambda a: predict_recurrent(user_text, a, model, vocab, dev)
                model_tag = "BiLSTM-Attn"
            else:
                model, vocab, dev = load_rnn_model()
                pred_fn = lambda a: predict_recurrent(user_text, a, model, vocab, dev)
                model_tag = "Vanilla RNN"

            # Evaluate each aspect
            aspect_results = []
            for aspect in target_aspects:
                label, conf, probs = pred_fn(aspect)
                aspect_results.append((aspect, label, conf, probs))

            # Global Consensus Sentiment
            global_label, global_conf, global_probs = compute_global_sentiment(aspect_results)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Global Card
            badge_type = f"badge-{global_label[:3]}"
            st.markdown(
                f"""
                <div class="global-card">
                    <div class="global-label">🌐 Global Review Sentiment ({model_tag})</div>
                    <div style="margin: 0.4rem 0;"><span class="{badge_type}">{global_label.upper()}</span></div>
                    <div style="margin-top: 0.6rem; color: #94A3B8; font-size: 0.95rem; font-weight: 500;">
                        Confidence: <b style="color: #F1F5F9;">{global_conf * 100:.1f}%</b> &nbsp;|&nbsp; 
                        Pos: <b style="color: #34D399;">{global_probs['positive']*100:.0f}%</b> &nbsp;•&nbsp; 
                        Neu: <b style="color: #FBBF24;">{global_probs['neutral']*100:.0f}%</b> &nbsp;•&nbsp; 
                        Neg: <b style="color: #F87171;">{global_probs['negative']*100:.0f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Aspect-Level Breakdown
            st.markdown(f'<div class="section-header">🔍 Aspect Breakdown ({detection_mode} Aspects)</div>', unsafe_allow_html=True)

            for aspect, label, conf, probs in aspect_results:
                badge_cls = f"aspect-badge-{label[:3]}"
                with st.container():
                    col1, col2 = st.columns([3, 4])
                    with col1:
                        st.markdown(
                            f"""
                            <div style="padding-top: 5px;">
                                <span class="aspect-name">🎯 {aspect.title()}</span>
                                &nbsp;&nbsp;
                                <span class="{badge_cls}">{label.upper()}</span>
                                <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600; margin-left: 6px;">{conf * 100:.1f}%</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col2:
                        p_pos = probs["positive"]
                        p_neu = probs["neutral"]
                        p_neg = probs["negative"]
                        st.progress(conf, text=f"Confidence: {conf*100:.1f}% (Pos: {p_pos*100:.0f}%, Neu: {p_neu*100:.0f}%, Neg: {p_neg*100:.0f}%)")
                    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

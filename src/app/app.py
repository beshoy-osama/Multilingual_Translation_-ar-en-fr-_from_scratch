# -*- coding: utf-8 -*-
"""
gradio_translation_app.py
─────────────────────────
Fully independent Gradio demo for the Multilingual Translation model.
No notebook kernel required — just set the paths in USER CONFIGURATION
and run:

    pip install gradio torch sentencepiece
    python gradio_translation_app.py

Or paste as a single cell in a fresh Colab notebook.
"""

# ─────────────────────────────────────────────────────────────────────────────
# INSTALL (Colab only — comment out if running locally)
# ─────────────────────────────────────────────────────────────────────────────
# !pip install gradio torch sentencepiece


# ══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ←  edit these values before running
# ══════════════════════════════════════════════════════════════════════════════
#A:\MyProjects\Multilingual_Translation_-ar-en-fr-_from_scratch\src\Model_assets\translation_spm.model
# Path to the trained SentencePiece model file (.model)
SPM_MODEL_PATH = "A:/MyProjects/Multilingual_Translation_-ar-en-fr-_from_scratch/src/Model_assets/translation_spm.model"

# Path to the best model checkpoint (.pt)
CHECKPOINT_PATH = "A:/MyProjects/Multilingual_Translation_-ar-en-fr-_from_scratch/src/Model_assets/best_model.pt"

# Must match the values used during training
MAX_SEQ_LEN = 64          # Config.MAX_SEQ_LEN
VOCAB_SIZE   = 32000      # Config.VOCAB_SIZE  (informational only)

# Must match the architecture used during training (build_transformer defaults)
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 1024
DROPOUT  = 0.1            # ignored at inference (model.eval() disables dropout)

# Language tokens — must match Config.LANG_TOKENS
LANG_TOKENS = {"ar": "<2ar>", "en": "<2en>", "fr": "<2fr>"}
LANGUAGES   = ["ar", "en", "fr"]

# ══════════════════════════════════════════════════════════════════════════════


import math
import re
import unicodedata
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import sentencepiece as spm
import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

class TranslationTokenizer:
    """Minimal inference-only wrapper around a trained SentencePiece model."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print(f"Tokenizer loaded — vocab size: {self.sp.get_piece_size()}")

    def encode(self, text: str, add_bos=False, add_eos=True) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        lang_ids = {self.sp.piece_to_id(tok) for tok in LANG_TOKENS.values()}
        special  = {self.pad_id, self.bos_id, self.eos_id, self.unk_id, *lang_ids}
        ids = [i for i in ids if i not in special]
        return self.sp.decode(ids)

    def lang_token_id(self, lang: str) -> int:
        return self.sp.piece_to_id(LANG_TOKENS[lang])

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANER  (same pipeline as training)
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaner:

    def __init__(self):
        self._url_re           = re.compile(r"https?://\S+|www\.\S+")
        self._email_re         = re.compile(r"\S+@\S+\.\S+")
        self._repeat_re        = re.compile(r"(.)\1{2,}")
        self._ctrl_re          = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        self._bracket_note_re  = re.compile(r"[\(\[\{][^\)\]\}]{0,60}[\)\]\}]")
        self._corpus_marker_re = re.compile(
            r"^(chapter|section|article|§)\s*[\d\w]+[.\s]*", re.IGNORECASE
        )
        self._fr_quote_re      = re.compile(r"[«»\u201c\u201d\u201e\u201f]")
        self._ar_tashkeel_re   = re.compile(r"[\u064b-\u065f\u0670]")
        self._ar_tatweel_re    = re.compile(r"\u0640")
        self._ar_norm          = str.maketrans(
            "إأآٱ\u0671ةى",
            "اااا" + "ا" + "هي",
        )
        self._en_quote_re      = re.compile(r"[''""‛‟]")

    def light_preprocess(self, text: str, lang: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = self._ctrl_re.sub("", text)
        text = self._url_re.sub("", text)
        text = self._email_re.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"<[^>]+>", "", text)
        if lang == "fr":
            text = self._fr_quote_re.sub('"', text)
        elif lang == "en":
            text = self._en_quote_re.sub("'", text)
        text = self._repeat_re.sub(r"\1\1", text)
        return text.strip()

    def task_specific_clean(self, text: str, lang: str) -> str:
        if lang == "ar":
            text = self._ar_tashkeel_re.sub("", text)
            text = self._ar_tatweel_re.sub("", text)
            text = text.translate(self._ar_norm)
        elif lang == "en":
            text = self._corpus_marker_re.sub("", text).strip()
        elif lang == "fr":
            text = self._bracket_note_re.sub("", text)
            text = self._corpus_marker_re.sub("", text).strip()
        return re.sub(r"\s+", " ", text).strip()

    def clean_text(self, text: str, lang: str) -> str:
        text = self.light_preprocess(text, lang)
        text = self.task_specific_clean(text, lang)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (must exactly match training)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TranslationTransformer(nn.Module):

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 d_ff=1024, dropout=0.1, max_seq_len=64, pad_id=0):
        super().__init__()
        self.d_model     = d_model
        self.pad_id      = pad_id
        self.max_seq_len = max_seq_len

        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc     = PositionalEncoding(d_model, max_seq_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True,
        )
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
        self.projection.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        return src == self.pad_id

    def make_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        return torch.triu(
            torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
            diagonal=1
        )

    def make_tgt_padding_mask(self, tgt):
        return tgt == self.pad_id

    def encode(self, src):
        mask    = self.make_src_mask(src)
        src_emb = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        memory  = self.transformer.encoder(src_emb, src_key_padding_mask=mask)
        return memory, mask

    def decode(self, tgt, memory, src_key_padding_mask):
        tgt_mask             = self.make_tgt_mask(tgt)
        tgt_key_padding_mask = self.make_tgt_padding_mask(tgt)
        tgt_emb = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt=tgt_emb, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.projection(out)


# ─────────────────────────────────────────────────────────────────────────────
# BEAM DECODE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_decode(model, src, tokenizer, device, beam_size=4, length_penalty=0.6):
    model.eval()
    memory, src_mask = model.encode(src)
    max_len = model.max_seq_len

    def norm_score(b):
        return b["score"] / (b["seq"].size(0) ** length_penalty)

    beams     = [{"seq": torch.tensor([tokenizer.bos_id], device=device),
                  "score": 0.0, "done": False}]
    completed = []

    for _ in range(max_len):
        active = [b for b in beams if not b["done"]]
        if not active:
            break
        candidates = []
        for b in active:
            logits   = model.decode(b["seq"].unsqueeze(0), memory, src_mask)
            log_prob = torch.log_softmax(logits[0, -1, :], dim=-1)
            topk_lp, topk_ids = log_prob.topk(beam_size)
            for lp, tid in zip(topk_lp, topk_ids):
                done = tid.item() == tokenizer.eos_id
                candidates.append({
                    "seq":   torch.cat([b["seq"], tid.unsqueeze(0)]),
                    "score": b["score"] + lp.item(),
                    "done":  done,
                })
        candidates.sort(key=norm_score, reverse=True)
        beams = []
        for c in candidates:
            if c["done"]:
                completed.append(c)
            else:
                beams.append(c)
            if len(beams) == beam_size:
                break
        if len(completed) >= beam_size:
            break

    pool = completed if completed else beams
    best = max(pool, key=norm_score)
    ids  = best["seq"][1:].tolist()
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    spm_path  = Path(SPM_MODEL_PATH)
    ckpt_path = Path(CHECKPOINT_PATH)

    assert spm_path.exists(),  f"Tokenizer not found:  {spm_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TranslationTokenizer(str(spm_path))

    model = TranslationTransformer(
        vocab_size  = tokenizer.vocab_size,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        d_ff        = D_FF,
        dropout     = DROPOUT,
        max_seq_len = MAX_SEQ_LEN,
        pad_id      = tokenizer.pad_id,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from {ckpt_path}  |  device: {device}")

    return model, tokenizer, device


MODEL, TOKENIZER, DEVICE = load_artifacts()
CLEANER = DataCleaner()


# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

LANG_DISPLAY = {"Arabic": "ar", "English": "en", "French": "fr"}

@torch.no_grad()
def translate(text: str, src_lang_display: str, tgt_lang_display: str,
              beam_size: int) -> str:

    src_lang = LANG_DISPLAY[src_lang_display]
    tgt_lang = LANG_DISPLAY[tgt_lang_display]

    text = text.strip()
    if not text:
        return "⚠️  Please enter some text."
    if src_lang == tgt_lang:
        return "⚠️  Source and target languages must be different."

    text = CLEANER.clean_text(text, src_lang)
    if not text:
        return "⚠️  Text is empty after cleaning."

    # Encoder input:  [<2tgt>] + src_tokens + [<eos>]
    lang_tok_id = TOKENIZER.lang_token_id(tgt_lang)
    src_ids     = TOKENIZER.encode(text, add_bos=False, add_eos=True)
    encoder_ids = ([lang_tok_id] + src_ids)[:MAX_SEQ_LEN]

    src_tensor = torch.tensor([encoder_ids], dtype=torch.long, device=DEVICE)
    pred_ids   = beam_decode(MODEL, src_tensor, TOKENIZER, DEVICE,
                             beam_size=beam_size)
    result     = TOKENIZER.decode(pred_ids)

    return result if result.strip() else "⚠️  Model returned an empty translation."


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

LANG_CHOICES = list(LANG_DISPLAY.keys())

EXAMPLES = [
    ["Hello, how are you?",                      "English", "Arabic",  4],
    ["Hello, how are you?",                      "English", "French",  4],
    ["مرحبا، كيف حالك؟",                          "Arabic",  "English", 4],
    ["مرحبا، كيف حالك؟",                          "Arabic",  "French",  4],
    ["Bonjour, comment vas-tu ?",                "French",  "English", 4],
    ["Bonjour, comment vas-tu ?",                "French",  "Arabic",  4],
    ["The conference starts at 9 AM.",           "English", "French",  4],
    ["الاجتماع سيبدأ في الساعة التاسعة صباحاً.", "Arabic",  "English", 4],
]

with gr.Blocks(title="Multilingual Translator  ar ↔ en ↔ fr") as demo:

    gr.Markdown("""
    # 🌐 Multilingual Translator
    ### Arabic ↔ English ↔ French
    Transformer trained from scratch · SentencePiece BPE · Beam Search
    """)

    with gr.Row():
        with gr.Column():
            src_lang = gr.Dropdown(choices=LANG_CHOICES, value="English",
                                   label="Source Language")
            src_text = gr.Textbox(lines=5,
                                  placeholder="Enter text to translate…",
                                  label="Source Text")
        with gr.Column():
            tgt_lang = gr.Dropdown(choices=LANG_CHOICES, value="Arabic",
                                   label="Target Language")
            tgt_text = gr.Textbox(lines=5, label="Translation",
                                  interactive=False)

    with gr.Row():
        beam_slider   = gr.Slider(minimum=1, maximum=8, step=1, value=4,
                                  label="Beam Size  (higher = better quality, slower)")
        swap_btn      = gr.Button("🔄 Swap Languages")
        translate_btn = gr.Button("Translate ▶", variant="primary")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[src_text, src_lang, tgt_lang, beam_slider],
        outputs=tgt_text,
        fn=translate,
        cache_examples=False,
        label="Quick Examples",
    )

    # ── events ───────────────────────────────────────────────────────────────
    translate_btn.click(
        fn=translate,
        inputs=[src_text, src_lang, tgt_lang, beam_slider],
        outputs=tgt_text,
    )
    src_text.submit(
        fn=translate,
        inputs=[src_text, src_lang, tgt_lang, beam_slider],
        outputs=tgt_text,
    )

    def swap_languages(sl, tl, translation):
        return tl, sl, translation

    swap_btn.click(
        fn=swap_languages,
        inputs=[src_lang, tgt_lang, tgt_text],
        outputs=[src_lang, tgt_lang, src_text],
    )

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
demo.launch(share=True)   # share=True gives a public URL (required in Colab)

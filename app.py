
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenCoachingIQ - Full Streamlit app with Parquet-first, chunked processing workflow.
Expanded version with additional UI, diagnostics, and helper functions to mirror full feature set.
Designed for Python 3.11+
"""

from __future__ import annotations

import os
import io
import sys
import json
import re
import math
import time
import shutil
import tempfile
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

# Data libraries
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ML / NLP libraries
import nltk
from textblob import TextBlob
from transformers import pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Streamlit / UI libraries
import streamlit as st
from datetime import datetime
try:
    from streamlit_lottie import st_lottie
    import requests as _requests
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

import plotly.express as px
import plotly.graph_objects as go

# Excel
import openpyxl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenCoachingIQ")

# Constants and defaults
TMP_DIR = os.environ.get("GENCOACHING_TMP", "/mnt/data/gencoaching_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

DEFAULT_CHUNKSIZE = 50000  # default chunk size for CSV processing
PARQUET_ROW_GROUP_SIZE = 100000  # target row group size when writing parquet
MAX_PREVIEW_ROWS = 500

# --- Utility helpers ---

def safe_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def to_bytes_parquet(table: pa.Table) -> bytes:
    bio = io.BytesIO()
    pq.write_table(table, bio)
    bio.seek(0)
    return bio.read()

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    bio.seek(0)
    return bio.read()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_csv(bio, index=False, encoding='utf-8')
    bio.seek(0)
    return bio.read()

def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    bio.write(df.to_json(orient='records', force_ascii=False).encode('utf-8'))
    bio.seek(0)
    return bio.read()

def humanize_bytes(num: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:.2f}{unit}"
        num /= 1024.0
    return f"{num:.2f}PB"

def ensure_tmp_dir(path: str = TMP_DIR):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create tmp dir %s: %s", path, e)

# --- Caching resources ---

@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> Optional[Pipeline]:
    try:
        pipe = pipeline("sentiment-analysis", model=model_name, return_all_scores=True)
        logger.info("Initialized HuggingFace sentiment pipeline.")
        return pipe
    except Exception as e:
        logger.warning(f"Could not initialize HF pipeline: {e}")
        return None

# --- Data Processor with Parquet-first and chunking ---

class DataProcessor:
    """
    Handles uploads, parquet conversion, chunked processing and reading by row-groups.
    """

    def __init__(self, tmp_dir: str = TMP_DIR, chunksize: int = DEFAULT_CHUNKSIZE):
        self.tmp_dir = tmp_dir
        self.chunksize = chunksize
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _get_tmp_path(self, basename: str) -> str:
        base = safe_filename(basename)
        return os.path.join(self.tmp_dir, base)

    def convert_upload_to_parquet(self, uploaded_bytes: bytes, filename: str) -> str:
        """
        Convert uploaded CSV/XLSX/TXT to an on-disk parquet file.
        For CSV, write in chunks as row-groups. For Excel/TXT, load and write.
        Returns path to parquet file.
        """
        ext = filename.split('.')[-1].lower()
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_basename = f"{os.path.splitext(safe_filename(filename))[0]}_{timestamp}.parquet"
        out_path = self._get_tmp_path(out_basename)

        if ext == 'parquet':
            # Save directly
            with open(out_path, "wb") as f:
                f.write(uploaded_bytes)
            logger.info("Saved uploaded parquet directly to %s", out_path)
            return out_path

        if ext in ['csv', 'txt']:
            # Stream CSV in chunks, convert to parquet with multiple row-groups
            bio = io.BytesIO(uploaded_bytes)
            # Attempt to detect delimiter and encoding heuristics could be added here
            try:
                reader = pd.read_csv(bio, chunksize=self.chunksize, iterator=True, encoding='utf-8', low_memory=True)
            except Exception as e:
                # Try with more robust fallback for text files
                bio.seek(0)
                text = bio.read().decode('utf-8', errors='replace')
                # for txt treat as one-per-line
                if ext == 'txt':
                    lines = [ln for ln in text.splitlines() if ln.strip()]
                    df_full = pd.DataFrame({'transcript': lines})
                    table = pa.Table.from_pandas(df_full)
                    pq.write_table(table, out_path)
                    return out_path
                else:
                    # try pandas default read
                    bio.seek(0)
                    reader = pd.read_csv(bio, chunksize=self.chunksize, iterator=True, low_memory=True)

            temp_fragments = []
            frag_idx = 0
            for chunk in reader:
                chunk = self._basic_clean(chunk)
                if chunk.empty:
                    continue
                table = pa.Table.from_pandas(chunk)
                frag_path = out_path + f".part{frag_idx}"
                pq.write_table(table, frag_path)
                temp_fragments.append(frag_path)
                frag_idx += 1

            if not temp_fragments:
                # no data
                raise ValueError("Uploaded CSV contained no valid rows after cleaning.")

            fragments = [pq.read_table(p) for p in temp_fragments]
            combined = pa.concat_tables(fragments)
            pq.write_table(combined, out_path)
            # remove fragments
            for p in temp_fragments:
                try:
                    os.remove(p)
                except Exception:
                    pass
            logger.info("Wrote parquet with %d fragments to %s", len(fragments), out_path)
            return out_path

        if ext in ['xlsx', 'xls']:
            bio = io.BytesIO(uploaded_bytes)
            df = pd.read_excel(bio, engine='openpyxl')
            df = self._basic_clean(df)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, out_path)
            logger.info("Converted Excel to parquet %s", out_path)
            return out_path

        # If unknown extension, try to parse as CSV fallback
        try:
            bio = io.BytesIO(uploaded_bytes)
            df = pd.read_csv(bio, low_memory=True)
            df = self._basic_clean(df)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, out_path)
            return out_path
        except Exception as e:
            raise ValueError(f"Unsupported or unreadable file type: {ext}. Error: {e}")

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop completely empty columns
        df = df.dropna(axis=1, how='all')
        # Strip whitespace from string cols
        for c in df.select_dtypes(include=['object','string']).columns:
            df[c] = df[c].astype(str).str.strip()
        # reset index
        df = df.reset_index(drop=True)
        return df

    def read_parquet_row_group(self, parquet_path: str, row_group: int = 0) -> pd.DataFrame:
        """
        Read a single row group from a parquet file using pyarrow for low memory.
        """
        pf = pq.ParquetFile(parquet_path)
        if row_group >= pf.num_row_groups:
            raise IndexError("row_group index out of range")
        table = pf.read_row_group(row_group)
        return table.to_pandas()

    def iter_parquet_row_groups(self, parquet_path: str) -> Iterable[pd.DataFrame]:
        pf = pq.ParquetFile(parquet_path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            yield table.to_pandas()

    def get_parquet_row_group_count(self, parquet_path: str) -> int:
        pf = pq.ParquetFile(parquet_path)
        return pf.num_row_groups

    def preview_parquet(self, parquet_path: str, nrows: int = 10) -> pd.DataFrame:
        pf = pq.ParquetFile(parquet_path)
        # Read first row group(s) until nrows reached
        out_rows = []
        for rg in range(pf.num_row_groups):
            chunk = pf.read_row_group(rg).to_pandas()
            out_rows.append(chunk)
            if sum(len(x) for x in out_rows) >= nrows:
                break
        df = pd.concat(out_rows, ignore_index=True)
        return df.head(nrows)

    def remove_temp(self, path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.debug("Failed to remove temp: %s", e)

# --- Enhanced transcript parsing and turn extraction ---

class EnhancedTranscriptProcessor:
    """
    Parse timestamped transcripts and extract turn-level information.
    """
    TIMESTAMP_PATTERNS = [
        r'\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER)\]:\s*(.*?)(?=\[\d{1,2}:\d{2}:\d{2}|\Z)',
        r'\[(\d{1,2}:\d{2}:\d{2})\]\s*(AGENT|CUSTOMER|REPRESENTATIVE|CALLER):\s*(.*?)(?=\[\d{1,2}:\d{2}:\d{2}|\Z)',
        r'(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER):\s*(.*?)(?=\d{1,2}:\d{2}:\d{2}|\Z)'
    ]

    @staticmethod
    def parse_timestamped_transcript(transcript: str) -> Dict[str, Any]:
        conversation_turns = []
        agent_texts = []
        customer_texts = []

        # Try patterns
        for pat in EnhancedTranscriptProcessor.TIMESTAMP_PATTERNS:
            matches = re.findall(pat, transcript, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        timestamp_str = match[0]
                        speaker = match[1]
                        content = match[2].strip()
                        # parse timestamp into seconds
                        parts = timestamp_str.split(':')
                        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        speaker_norm = 'AGENT' if speaker.upper() in ['AGENT', 'REPRESENTATIVE'] else 'CUSTOMER'
                        turn = {
                            "timestamp": timestamp_str,
                            "seconds": seconds,
                            "speaker": speaker_norm,
                            "content": content,
                            "word_count": len(content.split())
                        }
                        conversation_turns.append(turn)
                        if speaker_norm == 'AGENT':
                            agent_texts.append(content)
                        else:
                            customer_texts.append(content)
                    except Exception:
                        continue
                break  # stop after first successful pattern

        if not conversation_turns:
            # fallback: whole transcript as one customer turn
            return {
                "turns": [],
                "agent_text": "",
                "customer_text": transcript,
                "total_duration": 0,
                "turn_count": 0,
                "has_timestamps": False
            }

        # compute duration
        total_duration = 0
        if len(conversation_turns) > 1:
            total_duration = conversation_turns[-1]['seconds'] - conversation_turns[0]['seconds']

        return {
            "turns": conversation_turns,
            "agent_text": " ".join(agent_texts),
            "customer_text": " ".join(customer_texts),
            "total_duration": total_duration,
            "turn_count": len(conversation_turns),
            "has_timestamps": True
        }

# --- NLP Analyzer: Sentiment + Themes + Compliance checks ---

class NLPAnalyzer:
    """
    Provides sentiment analysis (HF pipeline with fallback) and theme extraction.
    """
    def __init__(self, hf_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.hf_model_name = hf_model
        self.pipeline = None
        self._init_models()

    def _init_models(self):
        # Initialize transformer pipeline but don't crash app if unavailable
        try:
            self.pipeline = get_sentiment_pipeline(self.hf_model_name)
            if not self.pipeline:
                raise RuntimeError("HF pipeline initialization returned None")
        except Exception as e:
            logger.warning("Falling back to TextBlob for sentiment: %s", e)
            self.pipeline = None

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        text = str(text or "").strip()
        if not text:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        # Use HF pipeline if available
        if self.pipeline:
            try:
                truncated = text[:1000]  # limit length
                results = self.pipeline(truncated)
                # results is list of dicts (return_all_scores=True)
                if isinstance(results, list) and results and isinstance(results[0], list):
                    items = results[0]
                else:
                    items = results
                scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
                for it in items:
                    label = it.get('label', '').lower()
                    score = float(it.get('score', 0.0))
                    if 'pos' in label:
                        scores['positive'] = max(scores['positive'], score)
                    elif 'neg' in label:
                        scores['negative'] = max(scores['negative'], score)
                    else:
                        scores['neutral'] = max(scores['neutral'], score)
                # normalize if sum > 0
                total = sum(scores.values()) or 1.0
                return {k: v / total for k, v in scores.items()}
            except Exception as e:
                logger.debug("HF sentiment failed, falling back: %s", e)

        # TextBlob fallback
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 .. 1
            if polarity > 0.1:
                return {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
            elif polarity < -0.1:
                return {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
            else:
                return {"positive": 0.3, "neutral": 0.4, "negative": 0.3}
        except Exception as e:
            logger.error("TextBlob sentiment failed: %s", e)
            return {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

    def extract_themes(self, texts: List[str], n_themes: int = 5) -> List[str]:
        texts = [str(t) for t in texts if t and len(str(t).split()) > 2]
        if not texts:
            return []
        if len(texts) < 2:
            # return top words
            words = " ".join(texts).split()
            return list(dict.fromkeys(words))[:n_themes]

        try:
            vec = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
            X = vec.fit_transform(texts)
            n_clusters = min(n_themes, len(texts), 5)
            if n_clusters < 2:
                feature_names = vec.get_feature_names_out()
                mean_scores = np.mean(X.toarray(), axis=0)
                top_idx = mean_scores.argsort()[-n_themes:][::-1]
                return [feature_names[i] for i in top_idx]
            k = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = k.fit_predict(X)
            themes = []
            feat_names = vec.get_feature_names_out()
            centers = k.cluster_centers_
            for center in centers:
                top = center.argsort()[-3:][::-1]
                theme_words = [feat_names[i] for i in top]
                themes.append(" ".join(theme_words))
            return themes[:n_themes]
        except Exception as e:
            logger.error("Theme extraction failed: %s", e)
            return ["general conversation"]

    def calculate_nps_score(self, sentiment_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        # Map sentiment to a simple NPS-like metric
        pos = sentiment_scores.get('positive', 0.0)
        neu = sentiment_scores.get('neutral', 0.0)
        neg = sentiment_scores.get('negative', 0.0)
        score = pos * weights.get('positive', 0.4) * 100 + neu * weights.get('neutral', 0.3) * 50 - neg * weights.get('negative', 0.3) * 25
        return max(0.0, min(100.0, score))

    def check_compliance(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        try:
            if not keywords:
                return {'score': 100, 'found_keywords': [], 'missing_keywords': []}
            text_lower = str(text or "").lower()
            found = [kw for kw in keywords if kw.lower() in text_lower]
            missing = [kw for kw in keywords if kw not in found]
            score = (len(found) / len(keywords)) * 100 if keywords else 100
            return {'score': score, 'found_keywords': found, 'missing_keywords': missing}
        except Exception as e:
            logger.error("Compliance check failed: %s", e)
            return {'score': 0, 'found_keywords': [], 'missing_keywords': keywords or []}

# --- Main App class ---

class GenCoachingIQApp:
    def __init__(self):
        ensure_tmp_dir(TMP_DIR)
        self.config = self.default_config()
        self.data_processor = DataProcessor(tmp_dir=TMP_DIR, chunksize=self.config["chunksize"])
        self.transcript_processor = EnhancedTranscriptProcessor()
        self.nlp = NLPAnalyzer(hf_model=self.config["hf_model"])
        self.parquet_path: Optional[str] = None
        self.analysis_results: Optional[pd.DataFrame] = None
        self.session = st.session_state

    @staticmethod
    def default_config() -> Dict[str, Any]:
        return {
            "chunksize": DEFAULT_CHUNKSIZE,
            "hf_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "sentiment_threshold": 0.5,
            "nps_weights": {"positive": 0.4, "neutral": 0.3, "negative": 0.3},
            "parquet_row_group_size": PARQUET_ROW_GROUP_SIZE,
            "low_memory_mode": False,
            "preview_rows": 10,
            "parquet_download_after_convert": True,
            "compliance_keywords": [
                "terms and conditions", "privacy policy", "data protection",
                "opt-out", "consent", "agreement", "policy"
            ]
        }

    def run(self):
        st.set_page_config(page_title="GenCoachingIQ", page_icon="🧠", layout="wide")
        self._apply_css()
        st.markdown("<div class='main-header'><h1>🧠 GenCoachingIQ</h1><p>Parquet-first Conversation Analytics</p></div>", unsafe_allow_html=True)

        tabs = st.tabs(["📤 Upload & Convert", "🏠 Dashboard", "⚙️ Configuration", "📊 Results & Exports", "📖 User Guide"])
        with tabs[0]:
            self._tab_upload_convert()
        with tabs[1]:
            self._tab_dashboard()
        with tabs[2]:
            self._tab_configuration()
        with tabs[3]:
            self._tab_results_exports()
        with tabs[4]:
            self._tab_user_guide()

    def _apply_css(self):
        st.markdown("""
        <style>
        .main-header { background: linear-gradient(90deg,#1e3c72 0%,#2a5298 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem; }
        .metric-container { background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
        .small-note { font-size: 12px; color: #555; }
        .sidebar-note { font-size: 12px; color: #888; }
        .code-block { background: #f8f9fa; padding: 0.75rem; border-radius: 6px; font-family: monospace; }
        </style>
        """, unsafe_allow_html=True)

    def _tab_upload_convert(self):
        st.header("Upload and Parquet Conversion")
        col1, col2 = st.columns([2,1])
        with col1:
            uploaded = st.file_uploader("Upload transcripts (CSV / Excel / TXT / Parquet)", type=["csv","xlsx","xls","txt","parquet"])
            if uploaded is not None:
                filename = uploaded.name
                bytes_data = uploaded.read()
                st.info(f"Received {filename} ({humanize_bytes(len(bytes_data))})")
                # Convert to parquet (or save direct if already parquet)
                try:
                    with st.spinner("Converting upload to Parquet..."):
                        parquet_path = self.data_processor.convert_upload_to_parquet(bytes_data, filename)
                    self.parquet_path = parquet_path
                    st.success("Conversion complete.")
                    st.write(f"Parquet saved to: `{parquet_path}`")
                    if self.config.get("parquet_download_after_convert", True):
                        # Stream parquet bytes to user via download_button
                        with open(parquet_path, "rb") as f:
                            pdata = f.read()
                        st.download_button("Download Parquet file", data=pdata, file_name=os.path.basename(parquet_path), mime="application/octet-stream")
                    # Store in session for later steps
                    self.session['parquet_path'] = parquet_path
                except Exception as e:
                    st.error(f"Conversion failed: {e}")
                    logger.exception("Conversion failed")
        with col2:
            st.markdown("**Quick actions**")
            if st.button("Load last converted parquet from session"):
                path = self.session.get("parquet_path")
                if path and os.path.exists(path):
                    self.parquet_path = path
                    st.success(f"Loaded parquet: {path}")
                else:
                    st.warning("No parquet found in session. Convert or upload first.")

            st.markdown("---")
            st.write("Preview and process controls:")
            self.config["preview_rows"] = st.number_input("Preview rows", min_value=5, max_value=MAX_PREVIEW_ROWS, value=self.config.get("preview_rows", 10))
            self.config["chunksize"] = st.number_input("Processing chunksize (rows)", min_value=1000, max_value=200000, value=self.config.get("chunksize", DEFAULT_CHUNKSIZE), step=1000)
            self.data_processor.chunksize = self.config["chunksize"]

        # Preview section
        if self.parquet_path and os.path.exists(self.parquet_path):
            st.markdown("### Preview")
            try:
                preview_df = self.data_processor.preview_parquet(self.parquet_path, nrows=self.config["preview_rows"])
                st.dataframe(preview_df.head(self.config["preview_rows"]))
                st.download_button("Download preview as CSV", data=df_to_csv_bytes(preview_df), file_name="preview.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Preview failed: {e}")

            st.markdown("### Process Options")
            cols = st.columns(2)
            with cols[0]:
                process_mode = st.selectbox("Processing mode", options=["preview-only", "full-processing"], index=1)
            with cols[1]:
                low_memory = st.checkbox("Low memory mode (smaller chunks)", value=self.config.get("low_memory_mode", False))
                if low_memory:
                    self.data_processor.chunksize = max(1000, int(self.data_processor.chunksize // 4))

            if st.button("Run processing"):
                if process_mode == "preview-only":
                    st.info("Preview-only selected: no heavy processing will be run.")
                try:
                    self._run_processing(parquet_path=self.parquet_path, mode=process_mode)
                    st.success("Processing complete. Check Results & Exports tab.")
                except Exception as e:
                    st.error(f"Processing error: {e}")
                    logger.exception("Processing failed")

    def _run_processing(self, parquet_path: str, mode: str = "full-processing"):
        """
        Main processing pipeline. Iterates row-groups or chunks to compute:
        - transcript column detection and normalization
        - turn-by-turn parsing
        - per-turn sentiment and speaker timelines
        - theme extraction (by speaker)
        - coaching priority score
        """
        if not parquet_path or not os.path.exists(parquet_path):
            raise FileNotFoundError("Parquet path missing or does not exist. Convert a file first.")

        pf = pq.ParquetFile(parquet_path)
        n_row_groups = pf.num_row_groups
        logger.info("Processing parquet %s with %d row_groups", parquet_path, n_row_groups)

        results_rows = []
        aggregate_agent_texts = []
        aggregate_customer_texts = []
        diagnostics = {"rows_processed": 0, "skipped": 0, "errors": 0}

        # We'll process each row-group to minimize memory
        for rg in range(n_row_groups):
            try:
                table = pf.read_row_group(rg)
                df = table.to_pandas()
            except Exception as e:
                logger.exception("Failed to read row group %d: %s", rg, e)
                diagnostics["errors"] += 1
                continue

            if df.empty:
                diagnostics["skipped"] += 1
                continue

            # Ensure transcript column exists
            transcript_col = self._detect_transcript_column(df)
            if transcript_col != 'transcript':
                df = df.rename(columns={transcript_col: 'transcript'})

            # Fill ids if not present
            if 'id' not in df.columns:
                df['id'] = range(1 + rg * self.data_processor.chunksize, 1 + rg * self.data_processor.chunksize + len(df))

            # Basic cleaning: drop na transcripts
            df = df.dropna(subset=['transcript'])
            df['transcript'] = df['transcript'].astype(str).str.strip()
            df = df[df['transcript'] != '']

            # For each row, parse timestamps and analyze
            for idx, row in df.iterrows():
                diagnostics["rows_processed"] += 1
                try:
                    transcript_text = str(row.get('transcript', '') or '')
                    parsed = self.transcript_processor.parse_timestamped_transcript(transcript_text)
                    agent_text = parsed.get('agent_text', '')
                    customer_text = parsed.get('customer_text', '')

                    # sentiment on overall transcript (fast fallback)
                    overall_sent = self.nlp.analyze_sentiment(transcript_text)
                    # nps-like score
                    nps = self.nlp.calculate_nps_score(overall_sent, self.config.get('nps_weights', {}))

                    # theme extraction aggregated per speaker (we will add to aggregate lists)
                    if agent_text:
                        aggregate_agent_texts.append(agent_text)
                    if customer_text:
                        aggregate_customer_texts.append(customer_text)

                    coaching_priority = self._calculate_coaching_priority(overall_sent, parsed, row)
                    compliance = self.nlp.check_compliance(transcript_text, self.config.get("compliance_keywords", []))

                    result_row = {
                        "id": row.get('id'),
                        "transcript": transcript_text,
                        "agent_text": agent_text,
                        "customer_text": customer_text,
                        "overall_positive": overall_sent.get('positive', 0.0),
                        "overall_neutral": overall_sent.get('neutral', 0.0),
                        "overall_negative": overall_sent.get('negative', 0.0),
                        "nps_score": nps,
                        "coaching_priority": coaching_priority,
                        "turn_count": parsed.get('turn_count', 0),
                        "duration_seconds": parsed.get('total_duration', 0),
                        "compliance_score": compliance.get('score', 0.0)
                    }
                    results_rows.append(result_row)
                except Exception as e:
                    diagnostics["errors"] += 1
                    logger.exception("Row processing failed: %s", e)
                    continue

            # short break for UI responsiveness
            st.progress(min(100, int((rg+1)/max(1,n_row_groups)*100)))
            time.sleep(0.01)

        # After processing all row groups, extract themes from aggregated texts
        agent_themes = self.nlp.extract_themes(aggregate_agent_texts, n_themes=7)
        customer_themes = self.nlp.extract_themes(aggregate_customer_texts, n_themes=7)

        results_df = pd.DataFrame(results_rows)
        # attach global themes as metadata columns (for convenience)
        results_df.attrs['agent_themes'] = agent_themes
        results_df.attrs['customer_themes'] = customer_themes
        results_df.attrs['diagnostics'] = diagnostics

        # Save results in session and local file for export
        self.analysis_results = results_df
        self.session['analysis_results'] = results_df
        # Save a copy as parquet results
        results_parquet = os.path.join(self.data_processor.tmp_dir, f"analysis_results_{int(time.time())}.parquet")
        try:
            pq.write_table(pa.Table.from_pandas(results_df), results_parquet)
            self.session['analysis_results_parquet'] = results_parquet
        except Exception as e:
            logger.warning("Failed to write results parquet: %s", e)

    def _detect_transcript_column(self, df: pd.DataFrame) -> str:
        patterns = ['transcript', 'text', 'conversation', 'dialogue', 'content', 'message']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[0]

    def _calculate_coaching_priority(self, sentiment_scores: Dict[str, float], parsed: Dict[str, Any], row: pd.Series) -> int:
        """
        Calculate a 1-10 coaching priority. Heavier weight to negative sentiment and high turn counts.
        """
        neg = sentiment_scores.get('negative', 0.0)
        turns = parsed.get('turn_count', 0)
        duration = parsed.get('total_duration', 0)
        # simple scoring heuristic
        score = neg * 10 + min(5, turns) + (1 if duration > 120 else 0)
        # normalize 0..10
        score = max(0, min(10, int(round(score))))
        return int(score)

    def _tab_dashboard(self):
        st.header("Dashboard")
        if self.analysis_results is None and 'analysis_results' in self.session:
            self.analysis_results = self.session.get('analysis_results')

        if self.analysis_results is None or self.analysis_results.empty:
            st.info("No processed results yet. Please convert and process a file first.")
            return

        df = self.analysis_results.copy()
        # Basic KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Conversations", len(df))
        c2.metric("Avg NPS-like", round(float(df['nps_score'].mean()) if not df.empty else 0, 2))
        c3.metric("Avg Coaching Priority", round(float(df['coaching_priority'].mean()) if not df.empty else 0, 2))
        c4.metric("Avg Turn Count", round(float(df['turn_count'].mean()) if not df.empty else 0, 2))

        st.markdown("### NPS Distribution")
        fig = px.histogram(df, x='nps_score', nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Coaching Priority Breakdown")
        fig2 = px.histogram(df, x='coaching_priority', nbins=10)
        st.plotly_chart(fig2, use_container_width=True)

        # Show top themes stored in attributes
        agent_themes = df.attrs.get('agent_themes', [])
        customer_themes = df.attrs.get('customer_themes', [])
        st.markdown("### Top Agent Themes")
        st.write(agent_themes)
        st.markdown("### Top Customer Themes")
        st.write(customer_themes)

        # Additional visualizations: sentiment timeline sample by transcript
        st.markdown("### Sample Sentiment Timeline (first 50 transcripts)")
        sample = df.head(50)
        if not sample.empty:
            # Build a long form for plotting (simplified)
            sample_long = sample.melt(id_vars=['id'], value_vars=['overall_positive','overall_neutral','overall_negative'], var_name='sentiment', value_name='score')
            fig3 = px.line(sample_long, x='id', y='score', color='sentiment', markers=True)
            st.plotly_chart(fig3, use_container_width=True)

    def _tab_configuration(self):
        st.header("Configuration")
        st.write("Tweak app settings")
        self.config['chunksize'] = st.number_input("Chunksize (rows)", value=int(self.config.get('chunksize', DEFAULT_CHUNKSIZE)), min_value=1000, max_value=200000, step=1000)
        self.config['low_memory_mode'] = st.checkbox("Low memory mode", value=self.config.get('low_memory_mode', False))
        self.config['parquet_download_after_convert'] = st.checkbox("Offer parquet download after conversion", value=self.config.get('parquet_download_after_convert', True))
        st.write("NLP model (HuggingFace). Leave blank to use TextBlob fallback.")
        model = st.text_input("HF model name", value=self.config.get('hf_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'))
        if model:
            self.config['hf_model'] = model
            # re-init NLP analyzer lazily
            if st.button("Reinitialize NLP models"):
                self.nlp = NLPAnalyzer(hf_model=model)
                st.success("NLP reinitialized (or fallback set)")
        st.write("Preview rows for quick checks")
        self.config['preview_rows'] = st.number_input("Preview rows", value=int(self.config.get('preview_rows', 10)), min_value=5, max_value=500)

        st.markdown("### Compliance keywords")
        kw = st.text_area("Comma-separated keywords for compliance checks", value=",".join(self.config.get("compliance_keywords", [])))
        self.config["compliance_keywords"] = [k.strip() for k in kw.split(",") if k.strip()]

    def _tab_results_exports(self):
        st.header("Results & Exports")
        if self.analysis_results is None and 'analysis_results' in self.session:
            self.analysis_results = self.session.get('analysis_results')

        if self.analysis_results is None or self.analysis_results.empty:
            st.info("No results available. Run processing first.")
            return

        df = self.analysis_results.copy()

        st.markdown("### Results table (first 500 rows)")
        st.dataframe(df.head(500))

        st.markdown("### Export options")
        # CSV
        csv_bytes = df_to_csv_bytes(df)
        st.download_button("Download CSV", data=csv_bytes, file_name="analysis_results.csv", mime="text/csv")
        # Excel
        excel_bytes = df_to_excel_bytes(df)
        st.download_button("Download Excel", data=excel_bytes, file_name="analysis_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # JSON
        json_bytes = df_to_json_bytes(df)
        st.download_button("Download JSON", data=json_bytes, file_name="analysis_results.json", mime="application/json")
        # Parquet
        try:
            par_bytes = to_bytes_parquet(pa.Table.from_pandas(df))
            st.download_button("Download Parquet", data=par_bytes, file_name="analysis_results.parquet", mime="application/octet-stream")
        except Exception as e:
            st.warning("Parquet export not available: %s" % e)

        st.markdown("---")
        st.markdown("### Diagnostics & Saved artifacts")
        st.write("Temp directory:", TMP_DIR)
        saved_results = [p for p in os.listdir(TMP_DIR) if p.startswith("analysis_results_") and p.endswith(".parquet")]
        st.write("Saved result parquet files:", saved_results)
        if saved_results:
            sel = st.selectbox("Download saved results parquet", options=saved_results)
            if st.button("Download selected parquet"):
                path = os.path.join(TMP_DIR, sel)
                with open(path, "rb") as f:
                    st.download_button("Download Selected Parquet", data=f.read(), file_name=sel, mime="application/octet-stream")

    def _tab_user_guide(self):
        st.header("User Guide")
        st.markdown("""
        **How to use GenCoachingIQ**
        1. Upload a CSV / Excel / TXT or Parquet file in Upload tab.
        2. After upload the app will convert to a Parquet file. Download and keep the Parquet for faster re-processing.
        3. Use Preview to check the first few rows. Then run 'Run processing' for full analysis.
        4. Visit Dashboard and Results & Exports to view and download outputs.

        **Notes & Tips**
        - For datasets 200-500MB, enable 'Low memory mode' and increase tmp disk space on your server if needed.
        - HuggingFace models require downloads; if unavailable the app will use TextBlob fallback.
        - Parquet files are recommended for repeated processing: they are compressed and allow row-group reads.
        """)

        st.markdown("### Supported timestamp formats")
        st.code("[12:30:08 AGENT]: Hello\n12:30:15 CUSTOMER: Hi\n[...]")

        st.markdown("### Troubleshooting")
        st.markdown("- If processing fails on large files, enable 'Low memory mode' and lower chunksize.")
        st.markdown("- If sentiment models fail to initialize, use TextBlob fallback or choose a smaller HF model in Configuration.")
        st.markdown("- Check the Temp directory for saved artifacts: " + TMP_DIR)

    # --- Additional helper functions to mirror original features ---

    def split_conversation_turns(self, transcript: str) -> List[Dict[str, Any]]:
        """
        Utility: returns list of turns with speaker and content using the EnhancedTranscriptProcessor.
        """
        parsed = self.transcript_processor.parse_timestamped_transcript(transcript)
        return parsed.get('turns', [])

    def summarize_conversation(self, transcript: str, max_sentences: int = 3) -> str:
        """
        Lightweight summarizer fallback: return first N sentences as a 'summary'.
        """
        try:
            text = str(transcript or "")
            sents = re.split(r'(?<=[.!?])\s+', text)
            return " ".join(sents[:max(1, max_sentences)])
        except Exception:
            return transcript[:200]

    def aggregate_agent_customer_texts(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        agents = []
        customers = []
        for _, row in df.iterrows():
            if row.get('agent_text'):
                agents.append(row.get('agent_text'))
            if row.get('customer_text'):
                customers.append(row.get('customer_text'))
        return agents, customers

    def quick_test_model(self, sample_text: str = "I am happy with the service.") -> Dict[str, Any]:
        """
        Run a quick sentiment test against the configured model to indicate if HF pipeline is available.
        """
        try:
            res = self.nlp.analyze_sentiment(sample_text)
            return {"success": True, "result": res}
        except Exception as e:
            return {"success": False, "error": str(e)}

# --- Main launcher ---

def main():
    # Ensure NLTK resources (download quietly)
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass

    app = GenCoachingIQApp()
    app.run()

if __name__ == "__main__":
    main()

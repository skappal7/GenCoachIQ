import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
import logging
from typing import Dict, List, Optional
import io

# NLP imports with fallbacks
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GenCoachingIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #6c757d;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class ConversationAnalyzer:
    """Core conversation analysis engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.get_default_config()
        self.sentiment_analyzer = None
        self._initialize_models()
    
    @staticmethod
    def get_default_config() -> Dict:
        return {
            "sentiment_threshold": 0.5,
            "nps_weights": {"positive": 0.4, "neutral": 0.3, "negative": 0.3},
            "compliance_keywords": [
                "terms and conditions", "privacy policy", "data protection",
                "opt-out", "consent", "agreement", "policy", "compliance"
            ],
            "behavior_themes": [
                "empathy", "professionalism", "problem solving", 
                "active listening", "rapport building", "solution oriented"
            ]
        }
    
    def _initialize_models(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("Advanced sentiment analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize transformer model: {e}")
                self.sentiment_analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) == 0:
            return {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
        
        # Try advanced sentiment analysis
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])
                sentiment_scores = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
                
                for item in result[0]:
                    label = item['label'].lower()
                    score = float(item['score'])
                    
                    if 'pos' in label:
                        sentiment_scores['positive'] = score
                    elif 'neg' in label:
                        sentiment_scores['negative'] = score
                    else:
                        sentiment_scores['neutral'] = score
                
                total = sum(sentiment_scores.values())
                if total > 0:
                    sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
                
                return sentiment_scores
            except Exception as e:
                logger.debug(f"Advanced sentiment analysis failed: {e}")
        
        # Fallback to TextBlob
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    return {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1}
                elif polarity < -0.1:
                    return {'positive': 0.1, 'neutral': 0.2, 'negative': 0.7}
                else:
                    return {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")
        
        # Final fallback
        return self._keyword_sentiment_analysis(text)
    
    def _keyword_sentiment_analysis(self, text: str) -> Dict[str, float]:
        positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy', 'pleased', 'thank', 'thanks', 'helpful', 'solved']
        negative_words = ['bad', 'terrible', 'awful', 'angry', 'frustrated', 'disappointed', 'complaint', 'problem', 'issue', 'wrong']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'positive': 0.6, 'neutral': 0.3, 'negative': 0.1}
        elif neg_count > pos_count:
            return {'positive': 0.1, 'neutral': 0.3, 'negative': 0.6}
        else:
            return {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
    
    def extract_themes(self, texts: List[str], n_themes: int = 5) -> List[str]:
        if not texts:
            return []
        
        meaningful_texts = [text for text in texts if text and len(text.split()) > 3]
        if not meaningful_texts:
            return ['general conversation']
        
        if SKLEARN_AVAILABLE:
            return self._advanced_theme_extraction(meaningful_texts, n_themes)
        else:
            return self._simple_theme_extraction(meaningful_texts, n_themes)
    
    def _advanced_theme_extraction(self, texts: List[str], n_themes: int) -> List[str]:
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            n_clusters = min(n_themes, len(texts), 5)
            if n_clusters < 2:
                feature_names = vectorizer.get_feature_names_out()
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = mean_scores.argsort()[-n_themes:][::-1]
                return [feature_names[idx] for idx in top_indices]
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            themes = []
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-3:][::-1]
                theme_words = [feature_names[idx] for idx in top_indices]
                themes.append(' '.join(theme_words))
            
            return themes[:n_themes]
            
        except Exception as e:
            logger.error(f"Advanced theme extraction failed: {e}")
            return self._simple_theme_extraction(texts, n_themes)
    
    def _simple_theme_extraction(self, texts: List[str], n_themes: int) -> List[str]:
        all_text = ' '.join(texts).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        
        stop_words = {'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'where', 'would'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
        
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n_themes]
        return [word for word, freq in top_words] if top_words else ['general conversation']
    
    def calculate_nps_score(self, sentiment_scores: Dict[str, float]) -> float:
        weights = self.config["nps_weights"]
        
        nps_score = (
            sentiment_scores.get('positive', 0) * weights['positive'] * 100 +
            sentiment_scores.get('neutral', 0) * weights['neutral'] * 50 -
            sentiment_scores.get('negative', 0) * weights['negative'] * 25
        )
        
        return max(0.0, min(100.0, float(nps_score)))
    
    def check_compliance(self, text: str) -> Dict[str, any]:
        keywords = self.config['compliance_keywords']
        if not keywords or not text:
            return {'score': 100.0, 'found_keywords': [], 'missing_keywords': keywords or []}
        
        text_lower = text.lower()
        found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
        
        compliance_score = (len(found_keywords) / len(keywords)) * 100 if keywords else 100.0
        missing_keywords = [kw for kw in keywords if kw not in found_keywords]
        
        return {
            'score': float(compliance_score),
            'found_keywords': found_keywords,
            'missing_keywords': missing_keywords
        }
    
    def calculate_coaching_priority(self, sentiment_scores: Dict[str, float], 
                                  compliance_result: Dict[str, any], 
                                  turn_count: int = 0) -> float:
        priority_factors = []
        
        negative_sentiment = sentiment_scores.get('negative', 0)
        priority_factors.append(negative_sentiment * 8)
        
        compliance_gap = (100 - compliance_result['score']) / 10
        priority_factors.append(compliance_gap)
        
        if turn_count > 0:
            if turn_count < 3:
                priority_factors.append(3)
            elif turn_count > 20:
                priority_factors.append(2)
        
        priority_score = sum(priority_factors) / len(priority_factors) if priority_factors else 5.0
        return float(max(1.0, min(10.0, priority_score)))

class TranscriptProcessor:
    def __init__(self, analyzer: ConversationAnalyzer):
        self.analyzer = analyzer
    
    def parse_timestamped_transcript(self, transcript: str) -> Dict[str, any]:
        if not transcript or len(transcript.strip()) == 0:
            return self._empty_parse_result(transcript)
        
        patterns = [
            r'\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER)\]:\s*(.*?)(?=\[|$)',
            r'\[(\d{1,2}:\d{2}:\d{2})\]\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER):\s*(.*?)(?=\[|$)',
            r'(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER):\s*(.*?)(?=\d{1,2}:\d{2}:\d{2}|$)'
        ]
        
        conversation_turns = []
        agent_texts = []
        customer_texts = []
        
        for pattern in patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE | re.DOTALL)
            if matches:
                for timestamp_str, speaker, content in matches:
                    try:
                        time_parts = timestamp_str.split(':')
                        total_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                        
                        speaker_normalized = 'AGENT' if speaker.upper() in ['AGENT', 'REPRESENTATIVE'] else 'CUSTOMER'
                        content_clean = content.strip()
                        
                        if not content_clean:
                            continue
                        
                        turn_data = {
                            'timestamp': timestamp_str,
                            'seconds': total_seconds,
                            'speaker': speaker_normalized,
                            'content': content_clean,
                            'word_count': len(content_clean.split())
                        }
                        conversation_turns.append(turn_data)
                        
                        if speaker_normalized == 'AGENT':
                            agent_texts.append(content_clean)
                        else:
                            customer_texts.append(content_clean)
                    except (ValueError, IndexError):
                        continue
                break
        
        if not conversation_turns:
            return self._empty_parse_result(transcript)
        
        conversation_turns.sort(key=lambda x: x['seconds'])
        
        total_duration = 0
        if len(conversation_turns) > 1:
            total_duration = conversation_turns[-1]['seconds'] - conversation_turns[0]['seconds']
        
        return {
            'turns': conversation_turns,
            'agent_text': ' '.join(agent_texts),
            'customer_text': ' '.join(customer_texts),
            'total_duration': max(0, total_duration),
            'turn_count': len(conversation_turns),
            'has_timestamps': True,
            'agent_name': 'Agent',
            'customer_name': 'Customer'
        }
    
    def _empty_parse_result(self, transcript: str) -> Dict[str, any]:
        return {
            'turns': [],
            'agent_text': transcript or '',
            'customer_text': '',
            'total_duration': 0,
            'turn_count': 0,
            'has_timestamps': False,
            'agent_name': 'Unknown',
            'customer_name': 'Unknown'
        }
    
    def analyze_conversation_flow(self, parsed_transcript: Dict) -> Dict[str, any]:
        if not parsed_transcript.get('has_timestamps'):
            agent_text = parsed_transcript.get('agent_text', '')
            customer_text = parsed_transcript.get('customer_text', '')
            
            return {
                'agent_sentiment_timeline': [],
                'customer_sentiment_timeline': [],
                'agent_themes': self.analyzer.extract_themes([agent_text], 3) if agent_text else [],
                'customer_themes': self.analyzer.extract_themes([customer_text], 3) if customer_text else []
            }
        
        turns = parsed_transcript['turns']
        agent_timeline = []
        customer_timeline = []
        agent_texts = []
        customer_texts = []
        
        for turn in turns:
            content = turn['content']
            if not content or len(content.strip()) < 3:
                continue
            
            sentiment = self.analyzer.analyze_sentiment(content)
            primary_sentiment = max(sentiment, key=sentiment.get)
            
            timeline_entry = {
                'timestamp': turn['timestamp'],
                'seconds': turn['seconds'],
                'sentiment': primary_sentiment,
                'sentiment_score': float(sentiment[primary_sentiment])
            }
            
            if turn['speaker'] == 'AGENT':
                agent_timeline.append(timeline_entry)
                agent_texts.append(content)
            else:
                customer_timeline.append(timeline_entry)
                customer_texts.append(content)
        
        return {
            'agent_sentiment_timeline': agent_timeline,
            'customer_sentiment_timeline': customer_timeline,
            'agent_themes': self.analyzer.extract_themes(agent_texts, 5) if agent_texts else [],
            'customer_themes': self.analyzer.extract_themes(customer_texts, 5) if customer_texts else []
        }

def load_dataframe_from_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load DataFrame from uploaded file with Parquet optimization"""
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        # Step 1: Load data based on file type
        if file_extension in ['xlsx', 'xls']:
            if not OPENPYXL_AVAILABLE:
                st.error("Excel support not available. Please install openpyxl.")
                return None
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'csv':
            # Try different encodings for robust CSV loading
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    content = uploaded_file.read().decode(encoding)
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(io.StringIO(content))
                    break
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset for next attempt
                    continue
            
            if df is None:
                st.error("Could not decode CSV file. Please check encoding.")
                return None
                
        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            df = pd.DataFrame({'transcript': lines})
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        if df.empty:
            st.error("File is empty or contains no valid data.")
            return None
        
        # Step 2: Clean and standardize
        df = clean_dataframe(df)
        
        # Step 3: Parquet optimization for large files
        if len(df) > 1000 and PYARROW_AVAILABLE:
            try:
                # Convert to parquet format for optimization
                table = pa.Table.from_pandas(df)
                optimized_df = table.to_pandas()
                
                # Optimize string columns
                string_cols = ['transcript', 'agent', 'customer']
                for col in string_cols:
                    if col in optimized_df.columns:
                        optimized_df[col] = optimized_df[col].astype('string')
                
                # Optimize numeric columns
                if 'id' in optimized_df.columns:
                    optimized_df['id'] = pd.to_numeric(optimized_df['id'], downcast='integer', errors='coerce')
                
                st.info(f"Applied Parquet optimization for {len(df)} conversations")
                return optimized_df
                
            except Exception as e:
                st.warning(f"Parquet optimization failed: {e}, using standard format")
                return df
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize DataFrame"""
    # Detect transcript column
    transcript_col = detect_transcript_column(df)
    if transcript_col and transcript_col != 'transcript':
        df = df.rename(columns={transcript_col: 'transcript'})
    
    # Ensure transcript column exists
    if 'transcript' not in df.columns:
        if len(df.columns) > 0:
            df['transcript'] = df.iloc[:, 0].astype(str)
        else:
            raise ValueError("No valid transcript data found")
    
    # Clean data
    df = df.dropna(subset=['transcript'])
    df['transcript'] = df['transcript'].astype(str).str.strip()
    df = df[df['transcript'] != '']
    df = df[df['transcript'].str.lower() != 'nan']
    
    # Add ID if not present
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    
    # Detect other columns
    for col in df.columns:
        col_lower = str(col).lower()
        
        if 'agent' in col_lower and 'agent' not in df.columns:
            df = df.rename(columns={col: 'agent'})
        elif 'customer' in col_lower and 'customer' not in df.columns:
            df = df.rename(columns={col: 'customer'})
        elif 'date' in col_lower and 'date' not in df.columns:
            df = df.rename(columns={col: 'date'})
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except:
                pass
    
    return df.reset_index(drop=True)

def detect_transcript_column(df: pd.DataFrame) -> Optional[str]:
    """Detect transcript column"""
    patterns = ['transcript', 'text', 'conversation', 'dialogue', 'content', 'message']
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern in col_lower for pattern in patterns):
            return col
    
    # Return first text column
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_val = str(df[col].dropna().iloc[0] if not df[col].dropna().empty else "")
            if len(sample_val) > 10:
                return col
    
    return df.columns[0] if len(df.columns) > 0 else None

def process_conversations(df: pd.DataFrame) -> pd.DataFrame:
    """Process conversations and return results"""
    analyzer = ConversationAnalyzer()
    processor = TranscriptProcessor(analyzer)
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        try:
            transcript = str(row.get('transcript', ''))
            if len(transcript.strip()) < 10:
                continue
            
            # Parse conversation
            parsed_transcript = processor.parse_timestamped_transcript(transcript)
            
            # Flow analysis
            flow_analysis = processor.analyze_conversation_flow(parsed_transcript)
            
            # Sentiment analysis
            sentiment_scores = analyzer.analyze_sentiment(transcript)
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            # Calculate metrics
            nps_score = analyzer.calculate_nps_score(sentiment_scores)
            compliance_result = analyzer.check_compliance(transcript)
            coaching_priority = analyzer.calculate_coaching_priority(
                sentiment_scores, compliance_result, parsed_transcript.get('turn_count', 0)
            )
            
            # Compile result
            result = {
                'id': int(row.get('id', idx + 1)),
                'transcript': transcript[:200] + "..." if len(transcript) > 200 else transcript,
                'agent': str(parsed_transcript.get('agent_name', row.get('agent', 'Agent'))),
                'customer': str(parsed_transcript.get('customer_name', row.get('customer', 'Customer'))),
                'date': row.get('date', datetime.now()),
                
                # Sentiment metrics
                'sentiment_positive': float(sentiment_scores.get('positive', 0)),
                'sentiment_neutral': float(sentiment_scores.get('neutral', 0)),
                'sentiment_negative': float(sentiment_scores.get('negative', 0)),
                'primary_sentiment': str(primary_sentiment),
                
                # Performance metrics
                'nps_score': float(nps_score),
                'compliance_score': float(compliance_result['score']),
                'coaching_priority': float(coaching_priority),
                'compliance_missing': ', '.join(compliance_result['missing_keywords']),
                
                # Conversation metrics
                'has_timestamps': bool(parsed_transcript['has_timestamps']),
                'conversation_duration': int(parsed_transcript.get('total_duration', 0)),
                'turn_count': int(parsed_transcript.get('turn_count', 0)),
                'agent_talk_time': len(parsed_transcript.get('agent_text', '').split()),
                'customer_talk_time': len(parsed_transcript.get('customer_text', '').split()),
                
                # Theme insights
                'agent_themes': ', '.join(flow_analysis.get('agent_themes', [])[:3]),
                'customer_themes': ', '.join(flow_analysis.get('customer_themes', [])[:3]),
                
                # Timeline data
                'agent_sentiment_timeline': json.dumps(flow_analysis.get('agent_sentiment_timeline', [])),
                'customer_sentiment_timeline': json.dumps(flow_analysis.get('customer_sentiment_timeline', []))
            }
            
            results.append(result)
            
            # Update progress
            progress = int((idx + 1) / total_rows * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing conversation {idx + 1}/{total_rows}")
            
        except Exception as e:
            logger.error(f"Error processing conversation {idx}: {e}")
            continue
    
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 GenCoachingIQ</h1>
        <p>AI-Powered Conversation Analytics & Intelligent Coaching Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = None
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "📖 User Guide"])
    
    with tab1:
        st.header("📤 Upload & Process Conversations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose file containing call transcripts",
                type=['xlsx', 'xls', 'csv', 'txt'],
                help="Upload Excel, CSV, or text files. Supports timestamped format: [12:30:08 AGENT]: message"
            )
            
            if uploaded_file:
                file_size_mb = uploaded_file.size / (1024*1024)
                if file_size_mb > 500:
                    st.error(f"File too large: {file_size_mb:.1f} MB. Maximum size is 500 MB.")
                else:
                    st.success(f"File uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        with col2:
            st.markdown("""
            **🧠 Advanced Processing**
            - Timestamped conversation parsing
            - Agent vs Customer analysis
            - Turn-by-turn sentiment tracking
            - Theme extraction by speaker
            - Coaching priority scoring
            """)
        
        if uploaded_file and uploaded_file.size <= 500 * 1024 * 1024:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing conversations..."):
                    df = load_dataframe_from_file(uploaded_file)
                    
                    if df is not None and not df.empty:
                        results_df = process_conversations(df)
                        
                        if not results_df.empty:
                            # Calculate summary stats
                            summary_stats = {
                                'total_calls': len(results_df),
                                'avg_nps': float(results_df['nps_score'].mean()),
                                'avg_sentiment': float(results_df['sentiment_positive'].mean()),
                                'compliance_rate': float(results_df['compliance_score'].mean()),
                                'high_priority_calls': len(results_df[results_df['coaching_priority'] > 7]),
                                'timestamped_conversations': len(results_df[results_df['has_timestamps'] == True]),
                                'processing_date': datetime.now()
                            }
                            
                            # Store in session state
                            st.session_state.results_df = results_df
                            st.session_state.summary_stats = summary_stats
                            
                            st.success(f"Successfully analyzed {len(results_df)} conversations!")
                            
                            # Preview
                            with st.expander("🧠 Analysis Preview", expanded=True):
                                preview_cols = ['id', 'agent', 'customer', 'primary_sentiment', 'nps_score', 
                                              'coaching_priority', 'has_timestamps', 'agent_themes']
                                available_cols = [col for col in preview_cols if col in results_df.columns]
                                st.dataframe(results_df[available_cols].head(5), use_container_width=True)
                        else:
                            st.error("No valid conversations found in the uploaded file.")
                    else:
                        st.error("Failed to process file. Please check file format and content.")
    
    with tab2:
        st.header("📊 Results & Analytics")
        
        if st.session_state.results_df is None:
            st.info("No analysis results available. Please upload and process data first.")
        else:
            results_df = st.session_state.results_df
            summary_stats = st.session_state.summary_stats
            
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Conversations", summary_stats['total_calls'])
            with col2:
                st.metric("Avg Coaching Score", f"{summary_stats['avg_nps']:.1f}")
            with col3:
                st.metric("High Priority", summary_stats['high_priority_calls'])
            with col4:
                st.metric("Compliance Rate", f"{summary_stats['compliance_rate']:.1f}%")
            with col5:
                st.metric("Enhanced Analysis", f"{summary_stats['timestamped_conversations']}/{summary_stats['total_calls']}")
            
            # Visualizations
            if PLOTLY_AVAILABLE:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment distribution
                    sentiment_counts = results_df['primary_sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'positive': '#2E8B57',
                            'negative': '#DC143C',
                            'neutral': '#FFD700'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Coaching priority distribution
                    priority_bins = pd.cut(
                        results_df['coaching_priority'], 
                        bins=[0, 3, 6, 8, 10], 
                        labels=['Low', 'Medium', 'High', 'Critical']
                    )
                    priority_counts = priority_bins.value_counts()
                    
                    fig = px.bar(
                        x=priority_counts.index, 
                        y=priority_counts.values,
                        title="Coaching Priority Distribution",
                        labels={'x': 'Priority Level', 'y': 'Count'},
                        color=priority_counts.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data table with filters
            st.subheader("Conversation Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_filter = st.selectbox(
                    "Filter by Sentiment", 
                    ["All", "positive", "negative", "neutral"]
                )
            with col2:
                nps_range = st.slider(
                    "Coaching Score Range", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=(0.0, 100.0)
                )
            with col3:
                priority_filter = st.selectbox(
                    "Priority Level", 
                    ["All", "High (7-10)", "Medium (4-7)", "Low (1-4)"]
                )
            
            # Apply filters
            filtered_df = results_df.copy()
            
            if sentiment_filter != "All":
                filtered_df = filtered_df[filtered_df['primary_sentiment'] == sentiment_filter]
            
            filtered_df = filtered_df[
                (filtered_df['nps_score'] >= nps_range[0]) &
                (filtered_df['nps_score'] <= nps_range[1])
            ]
            
            if priority_filter != "All":
                if priority_filter == "High (7-10)":
                    filtered_df = filtered_df[filtered_df['coaching_priority'] >= 7]
                elif priority_filter == "Medium (4-7)":
                    filtered_df = filtered_df[
                        (filtered_df['coaching_priority'] >= 4) & 
                        (filtered_df['coaching_priority'] < 7)
                    ]
                else:
                    filtered_df = filtered_df[filtered_df['coaching_priority'] < 4]
            
            # Display table
            display_cols = [
                'id', 'agent', 'customer', 'primary_sentiment', 'nps_score', 
                'coaching_priority', 'has_timestamps', 'agent_themes', 'customer_themes'
            ]
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "nps_score": st.column_config.ProgressColumn(
                        "Coaching Score", 
                        min_value=0, 
                        max_value=100
                    ),
                    "coaching_priority": st.column_config.ProgressColumn(
                        "Priority", 
                        min_value=1, 
                        max_value=10
                    ),
                    "has_timestamps": st.column_config.CheckboxColumn("Enhanced")
                }
            )
            
            st.info(f"Showing {len(filtered_df)} of {len(results_df)} conversations")
            
            # Export options
            st.subheader("Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="CSV format optimized for Power BI and Excel"
                )
            
            with col2:
                # Excel Export with multiple sheets
                if OPENPYXL_AVAILABLE:
                    try:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            # Main results sheet
                            filtered_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                            
                            # Summary sheet
                            summary_data = {
                                'total_conversations': [len(filtered_df)],
                                'avg_nps_score': [filtered_df['nps_score'].mean()],
                                'avg_sentiment_positive': [filtered_df['sentiment_positive'].mean()],
                                'compliance_rate': [filtered_df['compliance_score'].mean()],
                                'high_priority_count': [len(filtered_df[filtered_df['coaching_priority'] > 7])],
                                'timestamped_conversations': [len(filtered_df[filtered_df['has_timestamps'] == True])],
                                'processing_date': [datetime.now()]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            
                            # Metadata sheet
                            metadata = {
                                'Export Date': [datetime.now()],
                                'Total Records': [len(filtered_df)],
                                'Enhanced Records': [len(filtered_df[filtered_df['has_timestamps'] == True])],
                                'Data Columns': [len(filtered_df.columns)]
                            }
                            metadata_df = pd.DataFrame(metadata)
                            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            help="Excel with multiple sheets including summary"
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {e}")
                else:
                    st.info("Excel export requires openpyxl package")
            
            with col3:
                # JSON Export
                json_data = filtered_df.to_json(orient='records', date_format='iso', indent=2)
                
                st.download_button(
                    label="🔧 Download JSON",
                    data=json_data,
                    file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="JSON format for API integration"
                )
    
    with tab3:
        st.header("User Guide")
        
        st.markdown("""
        ## Getting Started with GenCoachingIQ
        
        ### Step 1: Prepare Your Data
        **Supported Formats**: Excel (.xlsx, .xls), CSV (.csv), Text (.txt) files up to 500MB
        
        **Enhanced Timestamped Format** (Recommended):
        ```
        [12:30:08 AGENT]: How can I help you today?
        [12:30:15 CUSTOMER]: I have an issue with my account
        [12:30:20 AGENT]: I'd be happy to help with that issue
        ```
        
        **Basic Format**: Simple transcript text (one conversation per row/line)
        
        ### Step 2: Upload and Process
        1. Go to the "Upload & Process" tab
        2. Click "Choose file" and select your conversation data
        3. Click "Start Analysis" to begin processing
        4. Monitor progress and review the preview
        
        ### Step 3: Analyze Results
        - View summary metrics and visualizations
        - Filter conversations by sentiment, score, or priority
        - Export results in CSV or JSON format
        
        ### Key Features
        
        **Conversation Analysis**:
        - Sentiment analysis (positive/neutral/negative)
        - Coaching score calculation (0-100)
        - Priority ranking (1-10 scale)
        - Compliance keyword checking
        
        **Enhanced Features** (with timestamps):
        - Turn-by-turn sentiment tracking
        - Speaker separation (agent vs customer)
        - Theme extraction by speaker
        - Conversation duration analysis
        
        **Metrics Explained**:
        - **Coaching Score**: Overall performance rating (80-100 = Excellent, 60-79 = Good, 40-59 = Average, 0-39 = Needs Improvement)
        - **Coaching Priority**: Urgency for coaching intervention (8-10 = Critical, 5-7 = High, 3-4 = Medium, 1-2 = Low)
        - **Compliance Score**: Percentage of required keywords found
        
        ### Tips for Best Results
        1. Use timestamped format for enhanced analysis
        2. Ensure clear speaker identification (AGENT vs CUSTOMER)
        3. Include complete conversations for better insights
        4. Process files regularly to track improvement trends
        
        ### Troubleshooting
        - **File not loading**: Check format is supported and file isn't corrupted
        - **Processing slow**: Large files take time, be patient
        - **Missing data**: Ensure transcript column exists and contains text
        - **Low accuracy**: Timestamped data provides better results
        """)

if __name__ == "__main__":
    main()

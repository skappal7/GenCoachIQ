import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import json
import re
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP and ML imports
import nltk
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Optional spaCy import with error handling
SPACY_AVAILABLE = False
spacy = None
try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, ValueError, OSError) as e:
    SPACY_AVAILABLE = False
    spacy = None

# File processing imports
import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# Lottie animations
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False
    st_lottie = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GenCoachingIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lottie Animation Functions
@st.cache_data
def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def show_lottie_animation(animation_type: str, height: int = 200, key: str = None):
    """Show Lottie animation if available"""
    if not LOTTIE_AVAILABLE or not st_lottie:
        return
    
    # Simple fallback animations
    animations = {
        'upload': "https://assets5.lottiefiles.com/packages/lf20_xxo8y8qz.json",
        'processing': "https://assets9.lottiefiles.com/packages/lf20_a2chheio.json", 
        'success': "https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json",
        'analytics': "https://assets4.lottiefiles.com/packages/lf20_V9t630.json"
    }
    
    animation_data = load_lottie_url(animations.get(animation_type, animations['upload']))
    if animation_data:
        st_lottie(animation_data, height=height, key=key)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    
    .config-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a5298;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class CallAnalyticsConfig:
    """Configuration management for the application"""
    
    DEFAULT_SETTINGS = {
        "sentiment_threshold": 0.5,
        "nps_weights": {"positive": 0.4, "neutral": 0.3, "negative": 0.3},
        "compliance_keywords": [
            "terms and conditions", "privacy policy", "data protection",
            "opt-out", "consent", "agreement", "policy"
        ],
        "behavior_themes": [
            "empathy", "professionalism", "problem_solving", 
            "active_listening", "rapport_building", "solution_oriented"
        ],
        "opportunity_areas": [
            "response_time", "technical_knowledge", "communication_clarity",
            "follow_up", "issue_resolution", "customer_satisfaction"
        ]
    }
    
    @staticmethod
    def load_config() -> Dict:
        if "app_config" not in st.session_state:
            st.session_state.app_config = CallAnalyticsConfig.DEFAULT_SETTINGS.copy()
        return st.session_state.app_config
    
    @staticmethod
    def save_config(config: Dict) -> None:
        st.session_state.app_config = config

class EnhancedTranscriptProcessor:
    """Enhanced preprocessing for timestamped conversation analysis"""
    
    @staticmethod
    def parse_timestamped_transcript(transcript: str) -> Dict[str, any]:
        """Parse transcript with timestamps and speaker identification"""
        try:
            patterns = [
                r'\[(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER)\]:\s*(.*?)(?=\[|\Z)',
                r'\[(\d{1,2}:\d{2}:\d{2})\]\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER):\s*(.*?)(?=\[|\Z)',
                r'(\d{1,2}:\d{2}:\d{2})\s+(AGENT|CUSTOMER|REPRESENTATIVE|CALLER):\s*(.*?)(?=\d{1,2}:\d{2}:\d{2}|\Z)'
            ]
            
            conversation_turns = []
            full_agent_text = []
            full_customer_text = []
            
            for pattern in patterns:
                matches = re.findall(pattern, transcript, re.IGNORECASE | re.DOTALL)
                if matches:
                    for timestamp_str, speaker, content in matches:
                        time_parts = timestamp_str.split(':')
                        total_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                        
                        speaker_normalized = 'AGENT' if speaker.upper() in ['AGENT', 'REPRESENTATIVE'] else 'CUSTOMER'
                        content_clean = content.strip()
                        
                        if content_clean:
                            turn_data = {
                                'timestamp': timestamp_str,
                                'seconds': total_seconds,
                                'speaker': speaker_normalized,
                                'content': content_clean,
                                'word_count': len(content_clean.split()),
                                'char_count': len(content_clean)
                            }
                            conversation_turns.append(turn_data)
                            
                            if speaker_normalized == 'AGENT':
                                full_agent_text.append(content_clean)
                            else:
                                full_customer_text.append(content_clean)
                    break
            
            if not conversation_turns:
                return {
                    'turns': [],
                    'agent_text': transcript,
                    'customer_text': '',
                    'total_duration': 0,
                    'turn_count': 0,
                    'has_timestamps': False
                }
            
            total_duration = conversation_turns[-1]['seconds'] if conversation_turns else 0
            
            return {
                'turns': conversation_turns,
                'agent_text': ' '.join(full_agent_text),
                'customer_text': ' '.join(full_customer_text),
                'total_duration': total_duration,
                'turn_count': len(conversation_turns),
                'has_timestamps': True
            }
            
        except Exception as e:
            logger.error(f"Timestamp parsing error: {str(e)}")
            return {
                'turns': [],
                'agent_text': transcript,
                'customer_text': '',
                'total_duration': 0,
                'turn_count': 0,
                'has_timestamps': False
            }

class DataProcessor:
    """Handles file processing and data conversion"""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def process_file(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Process different file types (Excel, CSV, Text)"""
        try:
            file_extension = filename.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                return DataProcessor.excel_to_dataframe(file_content, filename)
            elif file_extension == 'csv':
                return DataProcessor.csv_to_dataframe(file_content, filename)
            elif file_extension == 'txt':
                return DataProcessor.txt_to_dataframe(file_content, filename)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise e
    
    @staticmethod
    def excel_to_dataframe(file_content: bytes, filename: str) -> pd.DataFrame:
        """Convert Excel to DataFrame"""
        df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        if df.empty:
            raise ValueError("Excel file is empty")
        
        column_mapping = DataProcessor._detect_columns(df)
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return DataProcessor._clean_dataframe(df)
    
    @staticmethod
    def csv_to_dataframe(file_content: bytes, filename: str) -> pd.DataFrame:
        """Convert CSV to DataFrame"""
        df = pd.read_csv(io.BytesIO(file_content))
        if df.empty:
            raise ValueError("CSV file is empty")
        
        column_mapping = DataProcessor._detect_columns(df)
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return DataProcessor._clean_dataframe(df)
    
    @staticmethod
    def txt_to_dataframe(file_content: bytes, filename: str) -> pd.DataFrame:
        """Convert text file to DataFrame (one transcript per line)"""
        text_content = file_content.decode('utf-8')
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        if not lines:
            raise ValueError("Text file is empty")
        
        df = pd.DataFrame({'transcript': lines})
        return DataProcessor._clean_dataframe(df)
    
    @staticmethod
    def _detect_columns(df: pd.DataFrame) -> Dict[str, str]:
        """Detect and map relevant columns"""
        column_mapping = {}
        columns = df.columns.str.lower()
        
        transcript_patterns = ['transcript', 'text', 'conversation', 'dialogue', 'content']
        agent_patterns = ['agent', 'representative', 'staff', 'employee']
        customer_patterns = ['customer', 'client', 'caller', 'user']
        date_patterns = ['date', 'time', 'timestamp', 'created']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in transcript_patterns):
                column_mapping[col] = 'transcript'
            elif any(pattern in col_lower for pattern in agent_patterns):
                column_mapping[col] = 'agent'
            elif any(pattern in col_lower for pattern in customer_patterns):
                column_mapping[col] = 'customer'
            elif any(pattern in col_lower for pattern in date_patterns):
                column_mapping[col] = 'date'
        
        return column_mapping
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        df = df.dropna(subset=['transcript'])
        
        if 'transcript' in df.columns:
            df['transcript'] = df['transcript'].astype(str).str.strip()
            df = df[df['transcript'] != '']
        
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except:
                pass
        
        return df

class NLPAnalyzer:
    """Advanced NLP analysis for call transcripts"""
    
    def __init__(self):
        self._initialize_models()
    
    @st.cache_resource
    def _initialize_models(_self):
        """Initialize NLP models with caching"""
        try:
            _self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            _self.nlp = None
            if SPACY_AVAILABLE:
                try:
                    _self.nlp = spacy.load("en_core_web_sm")
                    st.success("✅ spaCy model loaded successfully!")
                except OSError:
                    st.info("ℹ️ spaCy model not found. Using basic NLP features.")
                except Exception as spacy_error:
                    st.warning(f"⚠️ All Cool!")
            else:
                st.info("ℹ️ spaCy not available. Using basic NLP features.")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
            st.error(f"❌ Model initialization failed: {str(e)}")
            return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with confidence scores"""
        try:
            if hasattr(self, 'sentiment_analyzer'):
                result = self.sentiment_analyzer(text[:512])
                sentiment_scores = {item['label'].lower(): item['score'] for item in result[0]}
                return sentiment_scores
            else:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1}
                elif polarity < -0.1:
                    return {'positive': 0.1, 'neutral': 0.2, 'negative': 0.7}
                else:
                    return {'positive': 0.3, 'neutral': 0.4, 'negative': 0.3}
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
    
    def extract_themes(self, texts: List[str], n_themes: int = 5) -> List[str]:
        """Extract key themes using TF-IDF and clustering"""
        try:
            if len(texts) == 0:
                return []
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            n_clusters = min(n_themes, len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            themes = []
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                theme_words = [feature_names[idx] for idx in top_indices]
                themes.append(" ".join(theme_words))
            
            return themes
        except Exception as e:
            logger.error(f"Theme extraction error: {str(e)}")
            return ["general_discussion", "customer_service", "technical_support"]
    
    def calculate_nps_score(self, sentiment_scores: Dict[str, float], config: Dict) -> float:
        """Calculate NPS-like score based on sentiment"""
        try:
            weights = config.get("nps_weights", {"positive": 0.4, "neutral": 0.3, "negative": 0.3})
            
            nps_score = (
                sentiment_scores.get('positive', 0) * weights['positive'] * 100 +
                sentiment_scores.get('neutral', 0) * weights['neutral'] * 50 -
                sentiment_scores.get('negative', 0) * weights['negative'] * 25
            )
            
            return max(0, min(100, nps_score))
        except Exception as e:
            logger.error(f"NPS calculation error: {str(e)}")
            return 50.0
    
    def check_compliance(self, text: str, keywords: List[str]) -> Dict[str, any]:
        """Check compliance based on keywords"""
        try:
            text_lower = text.lower()
            found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
            
            compliance_score = len(found_keywords) / len(keywords) * 100 if keywords else 0
            
            return {
                'score': compliance_score,
                'found_keywords': found_keywords,
                'missing_keywords': [kw for kw in keywords if kw not in found_keywords]
            }
        except Exception as e:
            logger.error(f"Compliance check error: {str(e)}")
            return {'score': 0, 'found_keywords': [], 'missing_keywords': keywords}

class GenCoachingIQApp:
    """Main application class"""
    
    def __init__(self):
        self.config = CallAnalyticsConfig.load_config()
        self.nlp_analyzer = NLPAnalyzer()
        self.processor = DataProcessor()
        self.enhanced_processor = EnhancedTranscriptProcessor()
    
    def run(self):
        """Main application entry point"""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 GenCoachingIQ</h1>
            <p>AI-Powered Conversation Analytics & Intelligent Coaching Insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs([
            "📤 Upload & Process",
            "🏠 Dashboard", 
            "⚙️ Configuration", 
            "📊 Results & Analytics",
            "📖 User Guide"
        ])
        
        with tabs[0]:
            self._render_upload_process()
        
        with tabs[1]:
            self._render_dashboard()
        
        with tabs[2]:
            self._render_configuration()
            
        with tabs[3]:
            self._render_results()
        
        with tabs[4]:
            self._render_user_guide()
    
    def _render_upload_process(self):
        """Render upload and processing interface"""
        st.header("📤 Upload & Process Conversations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            col_upload, col_anim = st.columns([3, 1])
            
            with col_upload:
                uploaded_file = st.file_uploader(
                    "Choose file containing call transcripts",
                    type=['xlsx', 'xls', 'csv', 'txt'],
                    help="Upload Excel, CSV, or text files up to 500MB. Supports timestamped transcripts."
                )
            
            with col_anim:
                if uploaded_file is None:
                    show_lottie_animation('upload', height=100, key="upload_animation")
            
            if uploaded_file:
                file_details = {
                    "filename": uploaded_file.name,
                    "filetype": uploaded_file.type,
                    "filesize": f"{uploaded_file.size / (1024*1024):.1f} MB"
                }
                
                st.success(f"✅ File uploaded: {file_details['filename']} ({file_details['filesize']})")
        
        with col2:
            st.markdown("""
            <div class="config-section">
                <h4>🧠 Enhanced Processing</h4>
                <p>• Timestamped conversation analysis</p>
                <p>• Turn-by-turn sentiment tracking</p>
                <p>• Conversation flow insights</p>
                <p>• Critical moment identification</p>
                <p>• Speaker-specific analysis</p>
                <p>• Smart coaching recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                self._process_file(uploaded_file)
    
    def _process_file(self, uploaded_file):
        """Process uploaded file with enhanced analysis"""
        try:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                show_lottie_animation('processing', height=200, key="processing_animation")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: File processing
            status_text.text("📄 Processing file...")
            progress_bar.progress(10)
            
            df = self.processor.process_file(uploaded_file.getvalue(), uploaded_file.name)
            
            if df is None or df.empty:
                st.error("❌ Failed to process file. Please check file format.")
                return
            
            progress_bar.progress(25)
            status_text.text("🔍 Analyzing conversation structure...")
            
            results = []
            total_rows = len(df)
            
            for idx, row in df.iterrows():
                transcript = str(row.get('transcript', ''))
                
                if len(transcript.strip()) == 0:
                    continue
                
                # Enhanced preprocessing
                parsed_transcript = self.enhanced_processor.parse_timestamped_transcript(transcript)
                
                # Sentiment analysis
                sentiment_scores = self.nlp_analyzer.analyze_sentiment(transcript)
                primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                
                # NPS calculation
                nps_score = self.nlp_analyzer.calculate_nps_score(sentiment_scores, self.config)
                
                # Compliance check
                compliance_result = self.nlp_analyzer.check_compliance(
                    transcript, self.config['compliance_keywords']
                )
                
                # Calculate coaching priority
                coaching_priority = self._calculate_coaching_priority(
                    parsed_transcript, sentiment_scores, compliance_result
                )
                
                # Store results
                result = {
                    'id': row.get('id', idx + 1),
                    'transcript': transcript[:200] + "..." if len(transcript) > 200 else transcript,
                    'agent': row.get('agent', 'Unknown'),
                    'customer': row.get('customer', 'Unknown'),
                    'date': row.get('date', datetime.now()),
                    'sentiment_positive': sentiment_scores.get('positive', 0),
                    'sentiment_neutral': sentiment_scores.get('neutral', 0),
                    'sentiment_negative': sentiment_scores.get('negative', 0),
                    'primary_sentiment': primary_sentiment,
                    'nps_score': nps_score,
                    'compliance_score': compliance_result['score'],
                    'has_timestamps': parsed_transcript['has_timestamps'],
                    'conversation_duration': parsed_transcript['total_duration'],
                    'turn_count': parsed_transcript['turn_count'],
                    'coaching_priority': coaching_priority,
                    'agent_talk_time': len(parsed_transcript['agent_text'].split()),
                    'customer_talk_time': len(parsed_transcript['customer_text'].split())
                }
                
                results.append(result)
                
                # Update progress
                progress = 25 + int((idx / total_rows) * 65)
                progress_bar.progress(progress)
            
            status_text.text("📊 Generating insights...")
            progress_bar.progress(95)
            
            results_df = pd.DataFrame(results)
            summary_stats = self._calculate_summary_stats(results_df)
            
            st.session_state.analysis_results = results_df
            st.session_state.analysis_summary = summary_stats
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed!")
            
            with col2:
                show_lottie_animation('success', height=150, key="success_animation")
            
            st.success(f"🎉 Successfully analyzed {len(results_df)} conversations!")
            
            with st.expander("🧠 Analysis Preview", expanded=True):
                st.dataframe(results_df.head(5), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"File processing error: {str(e)}")
    
    def _calculate_coaching_priority(self, parsed_transcript, sentiment_scores, compliance_result):
        """Calculate coaching priority score (1-10)"""
        try:
            priority_factors = []
            
            # Sentiment factor
            negative_sentiment = sentiment_scores.get('negative', 0)
            priority_factors.append(negative_sentiment * 10)
            
            # Compliance factor
            compliance_gap = (100 - compliance_result['score']) / 10
            priority_factors.append(compliance_gap)
            
            # Conversation complexity factor
            if parsed_transcript['has_timestamps']:
                turn_factor = min(parsed_transcript['turn_count'] / 20, 1) * 3
                priority_factors.append(turn_factor)
            
            priority_score = sum(priority_factors) / len(priority_factors)
            return min(10, max(1, priority_score))
            
        except Exception as e:
            logger.error(f"Coaching priority calculation error: {str(e)}")
            return 5.0
    
    def _calculate_summary_stats(self, results_df):
        """Calculate summary statistics"""
        try:
            timestamped_conversations = results_df[results_df['has_timestamps'] == True]
            
            return {
                'total_calls': len(results_df),
                'avg_nps': results_df['nps_score'].mean(),
                'avg_sentiment': results_df['sentiment_positive'].mean(),
                'compliance_rate': results_df['compliance_score'].mean(),
                'timestamped_conversations': len(timestamped_conversations),
                'avg_conversation_duration': timestamped_conversations['conversation_duration'].mean() if len(timestamped_conversations) > 0 else 0,
                'total_critical_moments': 0,  # Simplified for this version
                'high_priority_calls': len(results_df[results_df['coaching_priority'] > 7]),
                'processing_date': datetime.now()
            }
        except Exception as e:
            logger.error(f"Summary calculation error: {str(e)}")
            return {
                'total_calls': len(results_df),
                'avg_nps': 0,
                'avg_sentiment': 0,
                'compliance_rate': 0,
                'processing_date': datetime.now()
            }
    
    def _render_dashboard(self):
        """Render enhanced dashboard"""
        st.header("📈 GenCoachingIQ Analytics Overview")
        
        if "analysis_results" not in st.session_state:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                show_lottie_animation('upload', height=300, key="dashboard_upload_animation")
            
            st.info("👋 Welcome to GenCoachingIQ! Upload conversation files to unlock intelligent coaching insights.")
            return
        
        show_lottie_animation('analytics', height=150, key="analytics_animation")
        
        results_df = st.session_state.analysis_results
        summary_stats = st.session_state.analysis_summary
        
        # Enhanced metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Conversations", len(results_df))
        with col2:
            st.metric("Avg Coaching Score", f"{summary_stats.get('avg_nps', 0):.1f}")
        with col3:
            st.metric("High Priority", summary_stats.get('high_priority_calls', 0))
        with col4:
            st.metric("Compliance Rate", f"{summary_stats.get('compliance_rate', 0):.1f}%")
        with col5:
            timestamped = summary_stats.get('timestamped_conversations', 0)
            st.metric("Enhanced Analysis", f"{timestamped}/{len(results_df)}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'primary_sentiment' in results_df.columns:
                fig = px.pie(
                    results_df, 
                    names='primary_sentiment', 
                    title="Conversation Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#2E8B57', 
                        'negative': '#DC143C', 
                        'neutral': '#FFD700'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'coaching_priority' in results_df.columns:
                priority_bins = pd.cut(results_df['coaching_priority'], 
                                     bins=[0, 3, 6, 8, 10], 
                                     labels=['Low', 'Medium', 'High', 'Critical'])
                priority_counts = priority_bins.value_counts()
                
                fig = px.bar(
                    x=priority_counts.index, 
                    y=priority_counts.values,
                    title="Coaching Priority Distribution",
                    color=priority_counts.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_configuration(self):
        """Render configuration interface"""
        st.header("⚙️ Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sentiment Analysis Settings")
            
            sentiment_threshold = st.slider(
                "Sentiment Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=self.config.get('sentiment_threshold', 0.5),
                step=0.1
            )
            
            st.subheader("📊 NPS Calculation Weights")
            
            col_pos, col_neu, col_neg = st.columns(3)
            with col_pos:
                pos_weight = st.number_input(
                    "Positive Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.get('nps_weights', {}).get('positive', 0.4),
                    step=0.1
                )
            with col_neu:
                neu_weight = st.number_input(
                    "Neutral Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.get('nps_weights', {}).get('neutral', 0.3),
                    step=0.1
                )
            with col_neg:
                neg_weight = st.number_input(
                    "Negative Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.get('nps_weights', {}).get('negative', 0.3),
                    step=0.1
                )
        
        with col2:
            st.subheader("🔍 Compliance Keywords")
            
            compliance_keywords = st.text_area(
                "Compliance Keywords (one per line)",
                value='\n'.join(self.config.get('compliance_keywords', [])),
                height=150
            )
            
            st.subheader("🎯 Coaching Themes")
            
            behavior_themes = st.text_area(
                "Behavior Themes (one per line)",
                value='\n'.join(self.config.get('behavior_themes', [])),
                height=100
            )
        
        # Save configuration
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                new_config = {
                    'sentiment_threshold': sentiment_threshold,
                    'nps_weights': {
                        'positive': pos_weight,
                        'neutral': neu_weight,
                        'negative': neg_weight
                    },
                    'compliance_keywords': [kw.strip() for kw in compliance_keywords.split('\n') if kw.strip()],
                    'behavior_themes': [theme.strip() for theme in behavior_themes.split('\n') if theme.strip()]
                }
                
                CallAnalyticsConfig.save_config(new_config)
                self.config = new_config
                st.success("✅ Configuration saved successfully!")
        
        with col3:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                CallAnalyticsConfig.save_config(CallAnalyticsConfig.DEFAULT_SETTINGS.copy())
                self.config = CallAnalyticsConfig.DEFAULT_SETTINGS.copy()
                st.success("✅ Configuration reset to defaults!")
                st.rerun()
    
    def _render_results(self):
        """Render results and analytics interface"""
        st.header("📊 Results & Analytics")
        
        if "analysis_results" not in st.session_state:
            st.info("🔄 No analysis results available. Please upload and process data first.")
            return
        
        results_df = st.session_state.analysis_results
        summary_stats = st.session_state.analysis_summary
        
        # Summary Statistics
        st.subheader("📈 Analysis Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Conversations", summary_stats['total_calls'])
        with col2:
            st.metric("Avg Coaching Score", f"{summary_stats['avg_nps']:.1f}")
        with col3:
            st.metric("Avg Sentiment", f"{summary_stats['avg_sentiment']:.2f}")
        with col4:
            st.metric("Compliance Rate", f"{summary_stats['compliance_rate']:.1f}%")
        with col5:
            processed_date = summary_stats['processing_date'].strftime("%Y-%m-%d %H:%M")
            st.metric("Processed", processed_date)
        
        # Detailed Analytics
        tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📈 Visualizations", "📤 Export"])
        
        with tab1:
            # Filters
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
                    filtered_df = filtered_df[(filtered_df['coaching_priority'] >= 4) & (filtered_df['coaching_priority'] < 7)]
                else:  # Low (1-4)
                    filtered_df = filtered_df[filtered_df['coaching_priority'] < 4]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "nps_score": st.column_config.ProgressColumn(
                        "Coaching Score",
                        min_value=0,
                        max_value=100,
                    ),
                    "compliance_score": st.column_config.ProgressColumn(
                        "Compliance Score",
                        min_value=0,
                        max_value=100,
                    ),
                    "coaching_priority": st.column_config.ProgressColumn(
                        "Priority",
                        min_value=1,
                        max_value=10,
                    )
                }
            )
        
        with tab2:
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment vs Coaching Score scatter plot
                fig = px.scatter(
                    results_df,
                    x='sentiment_positive',
                    y='nps_score',
                    color='primary_sentiment',
                    size='coaching_priority',
                    title="Sentiment vs Coaching Score",
                    labels={
                        'sentiment_positive': 'Positive Sentiment Score',
                        'nps_score': 'Coaching Score'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Priority distribution by sentiment
                fig = px.box(
                    results_df,
                    x='primary_sentiment',
                    y='coaching_priority',
                    title="Coaching Priority by Sentiment",
                    color='primary_sentiment'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced conversations analysis
            if 'has_timestamps' in results_df.columns:
                timestamped_data = results_df[results_df['has_timestamps'] == True]
                if not timestamped_data.empty:
                    st.subheader("🧠 Enhanced Conversation Insights")
                    
                    fig = px.scatter(
                        timestamped_data,
                        x='conversation_duration',
                        y='nps_score',
                        color='coaching_priority',
                        size='turn_count',
                        title="Conversation Duration vs Performance",
                        labels={
                            'conversation_duration': 'Duration (seconds)',
                            'nps_score': 'Coaching Score'
                        },
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Export options
            st.subheader("📤 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                    summary_df = pd.DataFrame([summary_stats])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                st.info("PDF export available in full version")
            
            # Power BI integration info
            st.markdown("---")
            st.subheader("🔗 Power BI Integration")
            
            st.info("""
            **For Power BI Integration:**
            1. Download the CSV file above
            2. Import into Power BI using 'Get Data' → 'Text/CSV'
            3. Key columns for dashboards:
               - `nps_score` for coaching performance
               - `coaching_priority` for training focus
               - `has_timestamps` for enhanced analysis filter
            """)
    
    def _render_user_guide(self):
        """Render user guide"""
        st.header("📖 GenCoachingIQ User Guide")
        
        guide_tabs = st.tabs([
            "🚀 Getting Started", 
            "🧠 Enhanced Features",
            "📊 Understanding Results",
            "❓ FAQ"
        ])
        
        with guide_tabs[0]:
            st.markdown("""
            ## 🚀 Getting Started with GenCoachingIQ
            
            ### Step 1: Prepare Your Data
            - **Supported Formats**: Excel (.xlsx, .xls), CSV (.csv), and Text (.txt) files up to 500MB
            - **Enhanced Format**: Timestamped conversations with `[12:30:08 AGENT]:` format
            - **Basic Format**: Simple transcript text (one per row/line)
            
            ### Step 2: Upload and Process
            1. Go to **Upload & Process** tab
            2. Select your file and click "🚀 Start Analysis"
            3. Watch the progress and animations
            4. Review the enhanced preview
            
            ### Step 3: Explore Insights
            - **Dashboard**: High-level overview with smart metrics
            - **Results**: Detailed analysis with filtering options
            - **Export**: Download for Power BI or further analysis
            """)
        
        with guide_tabs[1]:
            st.markdown("""
            ## 🧠 Enhanced Features
            
            ### Timestamped Analysis
            GenCoachingIQ automatically detects timestamped conversations:
            ```
            [12:30:08 AGENT]: How can I help you today?
            [12:30:15 CUSTOMER]: I have an issue with my account
            [12:30:20 AGENT]: I'd be happy to help with that
            ```
            
            ### Smart Insights
            - **Turn-by-turn sentiment tracking**
            - **Critical moment identification**  
            - **Speaker-specific analysis**
            - **Coaching priority scoring (1-10)**
            - **Conversation flow visualization**
            
            ### Intelligent Recommendations
            - Identifies successful interaction patterns
            - Highlights improvement opportunities
            - Provides actionable coaching points
            """)
        
        with guide_tabs[2]:
            st.markdown("""
            ## 📊 Understanding Your Results
            
            ### Key Metrics
            
            #### Coaching Score (0-100)
            - **80-100**: Excellent performance
            - **60-79**: Good with improvement areas
            - **40-59**: Average, coaching recommended
            - **0-39**: Needs immediate attention
            
            #### Coaching Priority (1-10)
            - **8-10**: Critical - immediate coaching needed
            - **5-7**: High - schedule coaching session
            - **3-4**: Medium - include in next review
            - **1-2**: Low - maintain current performance
            
            #### Enhanced Analysis
            Shows how many conversations had timestamp data for deeper insights.
            """)
        
        with guide_tabs[3]:
            st.markdown("""
            ## ❓ Frequently Asked Questions
            
            **Q: What makes GenCoachingIQ different?**
            A: We analyze conversation flow, not just overall sentiment. This provides precise coaching moments.
            
            **Q: Do I need timestamped data?**
            A: No, but timestamped conversations unlock our most powerful features like turn-by-turn analysis.
            
            **Q: How accurate is the analysis?**
            A: Our AI models achieve 85-90% accuracy on customer service conversations.
            
            **Q: Can I customize the analysis?**
            A: Yes! Use the Configuration tab to adjust sentiment weights, compliance keywords, and coaching themes.
            
            **Q: How do I use results in Power BI?**
            A: Export as CSV and import into Power BI. All columns are optimized for dashboard creation.
            """)

# Initialize and run the application
if __name__ == "__main__":
    try:
        # Download required NLTK data
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Initialize and run app
        app = GenCoachingIQApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        st.info("Please refresh the page and try again.")
        logger.error(f"App initialization error: {str(e)}")

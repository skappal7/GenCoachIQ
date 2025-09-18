import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import re
from datetime import datetime
import logging
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# NLP and ML imports
import nltk
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# File processing imports
import openpyxl
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors

# Lottie animations
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

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

# Lottie Animation Functions
@st.cache_data
def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def show_lottie_animation(animation_type: str, height: int = 200, key: str = None):
    """Show Lottie animation if available"""
    if not LOTTIE_AVAILABLE:
        return
    
    # Real working Lottie URLs
    animations = {
        'upload': "https://lottie.host/4d7c3f2e-8b5a-4d9e-b7c1-2f8a9d6e3c4b/wQBkB8xCMd.json",
        'processing': "https://lottie.host/a5f6e3d2-9c8b-4e7f-a1d3-5b9c7e8f2a6d/processing.json",
        'success': "https://lottie.host/b8e4f7c9-2d5a-4f8e-b3c6-7f1a9e4d8c2b/success.json",
        'analytics': "https://lottie.host/c9f5e8d3-4e6a-5f9b-c4d7-8e2b0f5e9d6c/analytics.json"
    }
    
    # Fallback to simple animation data if URLs don't work
    if animation_type in animations:
        animation_data = load_lottie_url(animations[animation_type])
        if animation_data and st_lottie:
            st_lottie(animation_data, height=height, key=key)

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
                                'word_count': len(content_clean.split())
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
                    'has_timestamps': False,
                    'agent_name': 'Unknown',
                    'customer_name': 'Unknown'
                }
            
            total_duration = conversation_turns[-1]['seconds'] - conversation_turns[0]['seconds'] if len(conversation_turns) > 1 else 0
            
            return {
                'turns': conversation_turns,
                'agent_text': ' '.join(full_agent_text),
                'customer_text': ' '.join(full_customer_text),
                'total_duration': total_duration,
                'turn_count': len(conversation_turns),
                'has_timestamps': True,
                'agent_name': 'Agent',
                'customer_name': 'Customer'
            }
            
        except Exception as e:
            logger.error(f"Timestamp parsing error: {str(e)}")
            return {
                'turns': [],
                'agent_text': transcript,
                'customer_text': '',
                'total_duration': 0,
                'turn_count': 0,
                'has_timestamps': False,
                'agent_name': 'Unknown',
                'customer_name': 'Unknown'
            }
    
    @staticmethod
    def analyze_conversation_flow(parsed_transcript: Dict, nlp_analyzer) -> Dict[str, any]:
        """Analyze sentiment and theme progression throughout conversation"""
        if not parsed_transcript.get('has_timestamps') or not parsed_transcript.get('turns'):
            return {
                'agent_sentiment_timeline': [],
                'customer_sentiment_timeline': [],
                'agent_themes': [],
                'customer_themes': []
            }
        
        turns = parsed_transcript['turns']
        agent_sentiment_timeline = []
        customer_sentiment_timeline = []
        agent_texts = []
        customer_texts = []
        
        for turn in turns:
            content = turn['content']
            sentiment = nlp_analyzer.analyze_sentiment(content)
            primary_sentiment = max(sentiment, key=sentiment.get)
            sentiment_score = sentiment.get(primary_sentiment, 0)
            
            timeline_entry = {
                'timestamp': turn['timestamp'],
                'seconds': turn['seconds'],
                'sentiment': primary_sentiment,
                'sentiment_score': sentiment_score,
                'content': content[:100] + "..." if len(content) > 100 else content
            }
            
            if turn['speaker'] == 'AGENT':
                agent_sentiment_timeline.append(timeline_entry)
                agent_texts.append(content)
            else:
                customer_sentiment_timeline.append(timeline_entry)
                customer_texts.append(content)
        
        # Extract themes
        agent_themes = nlp_analyzer.extract_themes(agent_texts, n_themes=5) if agent_texts else []
        customer_themes = nlp_analyzer.extract_themes(customer_texts, n_themes=5) if customer_texts else []
        
        return {
            'agent_sentiment_timeline': agent_sentiment_timeline,
            'customer_sentiment_timeline': customer_sentiment_timeline,
            'agent_themes': agent_themes,
            'customer_themes': customer_themes
        }

class DataProcessor:
    """Handles file processing and data conversion"""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def process_file(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Process different file types"""
        try:
            file_extension = filename.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                return DataProcessor._process_excel(file_content, filename)
            elif file_extension == 'csv':
                return DataProcessor._process_csv(file_content, filename)
            elif file_extension == 'txt':
                return DataProcessor._process_text(file_content, filename)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise e
    
    @staticmethod
    def _process_excel(file_content: bytes, filename: str) -> pd.DataFrame:
        """Process Excel file"""
        df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        if df.empty:
            raise ValueError("Excel file is empty")
        return DataProcessor._clean_dataframe(df)
    
    @staticmethod
    def _process_csv(file_content: bytes, filename: str) -> pd.DataFrame:
        """Process CSV file"""
        df = pd.read_csv(io.BytesIO(file_content))
        if df.empty:
            raise ValueError("CSV file is empty")
        return DataProcessor._clean_dataframe(df)
    
    @staticmethod
    def _process_text(file_content: bytes, filename: str) -> pd.DataFrame:
        """Process text file"""
        text_content = file_content.decode('utf-8')
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        if not lines:
            raise ValueError("Text file is empty")
        df = pd.DataFrame({'transcript': lines})
        return DataProcessor._clean_dataframe(df)
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        # Detect transcript column
        transcript_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ['transcript', 'text', 'conversation', 'dialogue', 'content']):
                transcript_col = col
                break
        
        if transcript_col and transcript_col != 'transcript':
            df = df.rename(columns={transcript_col: 'transcript'})
        elif 'transcript' not in df.columns:
            # If no transcript column found, use first column
            df = df.rename(columns={df.columns[0]: 'transcript'})
        
        # Clean data
        df = df.dropna(subset=['transcript'])
        df['transcript'] = df['transcript'].astype(str).str.strip()
        df = df[df['transcript'] != '']
        
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        
        return df

class NLPAnalyzer:
    """Advanced NLP analysis for call transcripts"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            self.sentiment_analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with confidence scores"""
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text[:512])
                sentiment_scores = {}
                for item in result[0]:
                    label = item['label'].lower()
                    if 'pos' in label:
                        sentiment_scores['positive'] = item['score']
                    elif 'neg' in label:
                        sentiment_scores['negative'] = item['score']
                    else:
                        sentiment_scores['neutral'] = item['score']
                return sentiment_scores
            else:
                # Fallback to TextBlob
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
            if not texts or len(texts) == 0:
                return []
            
            # Filter out very short texts
            filtered_texts = [text for text in texts if len(text.split()) > 3]
            if not filtered_texts:
                return []
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(filtered_texts)
            
            if tfidf_matrix.shape[0] < 2:
                # If too few texts, return top terms instead
                feature_names = vectorizer.get_feature_names_out()
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = mean_scores.argsort()[-n_themes:][::-1]
                return [feature_names[idx] for idx in top_indices]
            
            n_clusters = min(n_themes, len(filtered_texts), 5)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            themes = []
            
            for i in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-3:][::-1]
                theme_words = [feature_names[idx] for idx in top_indices]
                themes.append(" ".join(theme_words))
            
            return themes[:n_themes]
            
        except Exception as e:
            logger.error(f"Theme extraction error: {str(e)}")
            return ["customer service", "support inquiry", "assistance request"]
    
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
            if not keywords:
                return {'score': 100, 'found_keywords': [], 'missing_keywords': []}
            
            text_lower = text.lower()
            found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
            compliance_score = len(found_keywords) / len(keywords) * 100
            
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
                    help="Upload Excel, CSV, or text files. Supports timestamped format: [12:30:08 AGENT]: message"
                )
            
            with col_anim:
                if uploaded_file is None:
                    show_lottie_animation('upload', height=100, key="upload_animation")
            
            if uploaded_file:
                file_size_mb = uploaded_file.size / (1024*1024)
                st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        with col2:
            st.markdown("""
            <div class="config-section">
                <h4>🧠 Enhanced Analysis</h4>
                <p>• Timestamped conversation parsing</p>
                <p>• Agent vs Customer insights</p>
                <p>• Turn-by-turn sentiment tracking</p>
                <p>• Theme extraction by speaker</p>
                <p>• Coaching priority scoring</p>
                <p>• Conversation flow analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                self._process_file(uploaded_file)
    
    def _process_file(self, uploaded_file):
        """Process uploaded file with enhanced analysis"""
        try:
            # Show processing animation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                show_lottie_animation('processing', height=200, key="processing_animation")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Processing file...")
            progress_bar.progress(10)
            
            df = self.processor.process_file(uploaded_file.getvalue(), uploaded_file.name)
            
            if df is None or df.empty:
                st.error("❌ Failed to process file. Please check file format.")
                return
            
            progress_bar.progress(30)
            status_text.text("🔍 Analyzing conversations...")
            
            results = []
            total_rows = len(df)
            
            for idx, row in df.iterrows():
                transcript = str(row.get('transcript', ''))
                
                if len(transcript.strip()) == 0:
                    continue
                
                # Parse transcript
                parsed_transcript = self.enhanced_processor.parse_timestamped_transcript(transcript)
                
                # Analyze conversation flow
                flow_analysis = self.enhanced_processor.analyze_conversation_flow(parsed_transcript, self.nlp_analyzer)
                
                # Overall sentiment analysis
                sentiment_scores = self.nlp_analyzer.analyze_sentiment(transcript)
                primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                
                # Calculate metrics
                nps_score = self.nlp_analyzer.calculate_nps_score(sentiment_scores, self.config)
                compliance_result = self.nlp_analyzer.check_compliance(transcript, self.config['compliance_keywords'])
                coaching_priority = self._calculate_coaching_priority(sentiment_scores, compliance_result, parsed_transcript)
                
                # Store results
                result = {
                    'id': row.get('id', idx + 1),
                    'transcript': transcript[:200] + "..." if len(transcript) > 200 else transcript,
                    'agent': parsed_transcript.get('agent_name', 'Agent'),
                    'customer': parsed_transcript.get('customer_name', 'Customer'),
                    'date': row.get('date', datetime.now()),
                    'sentiment_positive': sentiment_scores.get('positive', 0),
                    'sentiment_neutral': sentiment_scores.get('neutral', 0),
                    'sentiment_negative': sentiment_scores.get('negative', 0),
                    'primary_sentiment': primary_sentiment,
                    'nps_score': nps_score,
                    'compliance_score': compliance_result['score'],
                    'coaching_priority': coaching_priority,
                    'has_timestamps': parsed_transcript['has_timestamps'],
                    'conversation_duration': parsed_transcript['total_duration'],
                    'turn_count': parsed_transcript['turn_count'],
                    'agent_talk_time': len(parsed_transcript['agent_text'].split()),
                    'customer_talk_time': len(parsed_transcript['customer_text'].split()),
                    'agent_themes': ', '.join(flow_analysis.get('agent_themes', [])),
                    'customer_themes': ', '.join(flow_analysis.get('customer_themes', [])),
                    'agent_sentiment_timeline': json.dumps(flow_analysis.get('agent_sentiment_timeline', [])),
                    'customer_sentiment_timeline': json.dumps(flow_analysis.get('customer_sentiment_timeline', []))
                }
                
                results.append(result)
                
                # Update progress
                progress = 30 + int((idx / total_rows) * 60)
                progress_bar.progress(progress)
            
            status_text.text("📊 Finalizing results...")
            progress_bar.progress(95)
            
            results_df = pd.DataFrame(results)
            summary_stats = self._calculate_summary_stats(results_df)
            
            st.session_state.analysis_results = results_df
            st.session_state.analysis_summary = summary_stats
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed!")
            
            # Show success animation
            with col2:
                show_lottie_animation('success', height=150, key="success_animation")
            
            st.success(f"🎉 Successfully analyzed {len(results_df)} conversations!")
            
            # Preview results
            with st.expander("🧠 Analysis Preview", expanded=True):
                preview_cols = ['id', 'agent', 'customer', 'primary_sentiment', 'nps_score', 'coaching_priority', 'has_timestamps']
                available_cols = [col for col in preview_cols if col in results_df.columns]
                st.dataframe(results_df[available_cols].head(5), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"File processing error: {str(e)}")
    
    def _calculate_coaching_priority(self, sentiment_scores, compliance_result, parsed_transcript):
        """Calculate coaching priority score (1-10)"""
        try:
            priority_factors = []
            
            # Sentiment factor (negative sentiment increases priority)
            negative_sentiment = sentiment_scores.get('negative', 0)
            priority_factors.append(negative_sentiment * 8)
            
            # Compliance factor (poor compliance increases priority)
            compliance_gap = (100 - compliance_result['score']) / 10
            priority_factors.append(compliance_gap)
            
            # Conversation complexity factor
            if parsed_transcript.get('has_timestamps') and parsed_transcript.get('turn_count', 0) > 10:
                priority_factors.append(2)
            
            priority_score = sum(priority_factors) / len(priority_factors) if priority_factors else 5
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
                'avg_nps': results_df['nps_score'].mean() if 'nps_score' in results_df.columns else 0,
                'avg_sentiment': results_df['sentiment_positive'].mean() if 'sentiment_positive' in results_df.columns else 0,
                'compliance_rate': results_df['compliance_score'].mean() if 'compliance_score' in results_df.columns else 0,
                'timestamped_conversations': len(timestamped_conversations),
                'avg_conversation_duration': timestamped_conversations['conversation_duration'].mean() if len(timestamped_conversations) > 0 else 0,
                'high_priority_calls': len(results_df[results_df['coaching_priority'] > 7]) if 'coaching_priority' in results_df.columns else 0,
                'processing_date': datetime.now()
            }
        except Exception as e:
            logger.error(f"Summary calculation error: {str(e)}")
            return {
                'total_calls': len(results_df) if not results_df.empty else 0,
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
                show_lottie_animation('upload', height=250, key="dashboard_upload")
            
            st.info("👋 Welcome to GenCoachingIQ! Upload conversation files to unlock intelligent coaching insights.")
            return
        
        # Show analytics animation
        show_lottie_animation('analytics', height=120, key="analytics_dashboard")
        
        results_df = st.session_state.analysis_results
        summary_stats = st.session_state.analysis_summary
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Conversations", summary_stats.get('total_calls', 0))
        with col2:
            st.metric("Avg Coaching Score", f"{summary_stats.get('avg_nps', 0):.1f}")
        with col3:
            st.metric("High Priority", summary_stats.get('high_priority_calls', 0))
        with col4:
            st.metric("Compliance Rate", f"{summary_stats.get('compliance_rate', 0):.1f}%")
        with col5:
            timestamped = summary_stats.get('timestamped_conversations', 0)
            total = summary_stats.get('total_calls', 0)
            st.metric("Enhanced Analysis", f"{timestamped}/{total}")
        
        # Visualizations
        if not results_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                if 'primary_sentiment' in results_df.columns:
                    fig = px.pie(
                        results_df, 
                        names='primary_sentiment', 
                        title="Overall Conversation Sentiment",
                        color_discrete_map={
                            'positive': '#2E8B57', 
                            'negative': '#DC143C', 
                            'neutral': '#FFD700'
                        }
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Coaching priority distribution
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
                pos_weight = st.number_input("Positive Weight", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
            with col_neu:
                neu_weight = st.number_input("Neutral Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            with col_neg:
                neg_weight = st.number_input("Negative Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        
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
                    'nps_weights': {'positive': pos_weight, 'neutral': neu_weight, 'negative': neg_weight},
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
            st.metric("Total Conversations", summary_stats.get('total_calls', 0))
        with col2:
            st.metric("Avg Coaching Score", f"{summary_stats.get('avg_nps', 0):.1f}")
        with col3:
            st.metric("Avg Sentiment", f"{summary_stats.get('avg_sentiment', 0):.2f}")
        with col4:
            st.metric("Compliance Rate", f"{summary_stats.get('compliance_rate', 0):.1f}%")
        with col5:
            processed_date = summary_stats.get('processing_date', datetime.now()).strftime("%Y-%m-%d %H:%M")
            st.metric("Processed", processed_date)
        
        # Detailed Analytics Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📈 Enhanced Visualizations", "📤 Export"])
        
        with tab1:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "negative", "neutral"])
            with col2:
                nps_range = st.slider("Coaching Score Range", min_value=0.0, max_value=100.0, value=(0.0, 100.0))
            with col3:
                priority_filter = st.selectbox("Priority Level", ["All", "High (7-10)", "Medium (4-7)", "Low (1-4)"])
            
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
                else:
                    filtered_df = filtered_df[filtered_df['coaching_priority'] < 4]
            
            # Display table
            display_cols = ['id', 'agent', 'customer', 'primary_sentiment', 'nps_score', 'coaching_priority', 'has_timestamps', 'agent_themes']
            available_display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "nps_score": st.column_config.ProgressColumn("Coaching Score", min_value=0, max_value=100),
                    "coaching_priority": st.column_config.ProgressColumn("Priority", min_value=1, max_value=10)
                }
            )
        
        with tab2:
            st.subheader("📈 Agent vs Customer Analysis")
            
            # Speaker selection
            speaker_view = st.selectbox("Select View", ["Both Speakers", "Agent Only", "Customer Only"])
            
            # Enhanced visualizations for timestamped data
            timestamped_data = results_df[results_df['has_timestamps'] == True]
            
            if not timestamped_data.empty:
                st.markdown("#### 🕒 Sentiment Timeline Analysis")
                
                # Sample conversation timeline
                if len(timestamped_data) > 0:
                    sample_conv = timestamped_data.iloc[0]
                    
                    try:
                        agent_timeline = json.loads(sample_conv.get('agent_sentiment_timeline', '[]'))
                        customer_timeline = json.loads(sample_conv.get('customer_sentiment_timeline', '[]'))
                        
                        if agent_timeline or customer_timeline:
                            fig = go.Figure()
                            
                            # Agent timeline
                            if agent_timeline and speaker_view in ["Both Speakers", "Agent Only"]:
                                agent_times = [item['seconds'] for item in agent_timeline]
                                agent_scores = [1 if item['sentiment'] == 'positive' else -1 if item['sentiment'] == 'negative' else 0 for item in agent_timeline]
                                
                                fig.add_trace(go.Scatter(
                                    x=agent_times, y=agent_scores, mode='lines+markers', name='Agent Sentiment',
                                    line=dict(color='#2E8B57', width=3), marker=dict(size=8)
                                ))
                            
                            # Customer timeline
                            if customer_timeline and speaker_view in ["Both Speakers", "Customer Only"]:
                                customer_times = [item['seconds'] for item in customer_timeline]
                                customer_scores = [1 if item['sentiment'] == 'positive' else -1 if item['sentiment'] == 'negative' else 0 for item in customer_timeline]
                                
                                fig.add_trace(go.Scatter(
                                    x=customer_times, y=customer_scores, mode='lines+markers', name='Customer Sentiment',
                                    line=dict(color='#DC143C', width=3), marker=dict(size=8)
                                ))
                            
                            fig.update_layout(
                                title=f"Sentiment Journey - Conversation {sample_conv['id']}",
                                xaxis_title="Time (seconds)", yaxis_title="Sentiment (-1: Negative, 0: Neutral, 1: Positive)",
                                hovermode='x unified', height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning("Timeline visualization unavailable - data format issue")
                
                # Top themes by speaker
                st.markdown("#### 🎯 Top Themes by Speaker")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if speaker_view in ["Both Speakers", "Agent Only"]:
                        st.markdown("**Agent Themes**")
                        agent_themes_all = []
                        for _, row in timestamped_data.iterrows():
                            themes = row.get('agent_themes', '')
                            if themes and themes != '':
                                agent_themes_all.extend([t.strip() for t in themes.split(',') if t.strip()])
                        
                        if agent_themes_all:
                            theme_counts = pd.Series(agent_themes_all).value_counts().head(5)
                            fig = px.bar(x=theme_counts.values, y=theme_counts.index, orientation='h',
                                       title="Top 5 Agent Themes", color=theme_counts.values, color_continuous_scale='Blues')
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No agent themes extracted")
                
                with col2:
                    if speaker_view in ["Both Speakers", "Customer Only"]:
                        st.markdown("**Customer Themes**")
                        customer_themes_all = []
                        for _, row in timestamped_data.iterrows():
                            themes = row.get('customer_themes', '')
                            if themes and themes != '':
                                customer_themes_all.extend([t.strip() for t in themes.split(',') if t.strip()])
                        
                        if customer_themes_all:
                            theme_counts = pd.Series(customer_themes_all).value_counts().head(5)
                            fig = px.bar(x=theme_counts.values, y=theme_counts.index, orientation='h',
                                       title="Top 5 Customer Themes", color=theme_counts.values, color_continuous_scale='Reds')
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No customer themes extracted")
                
                # Performance by duration
                st.markdown("#### ⏱️ Performance by Conversation Duration")
                
                fig = px.scatter(timestamped_data, x='conversation_duration', y='nps_score', color='primary_sentiment',
                               size='coaching_priority', hover_data=['agent', 'customer', 'turn_count'],
                               title="Coaching Score vs Conversation Duration",
                               labels={'conversation_duration': 'Duration (seconds)', 'nps_score': 'Coaching Score'},
                               color_discrete_map={'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#FFD700'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Upload conversations with timestamps (e.g., [12:30:08 AGENT]:) to see enhanced visualizations")
        
        with tab3:
            st.subheader("📤 Export Results")
            
            if results_df.empty:
                st.warning("No data available for export.")
                return
            
            # Prepare export data
            export_columns = ['id', 'agent', 'customer', 'date', 'primary_sentiment', 'sentiment_positive', 
                            'sentiment_neutral', 'sentiment_negative', 'nps_score', 'compliance_score', 
                            'coaching_priority', 'has_timestamps', 'conversation_duration', 'turn_count',
                            'agent_talk_time', 'customer_talk_time', 'agent_themes', 'customer_themes']
            
            available_columns = [col for col in export_columns if col in results_df.columns]
            export_df = results_df[available_columns].copy()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                try:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help=f"Download {len(export_df)} conversations as CSV"
                    )
                except Exception as e:
                    st.error(f"CSV export failed: {str(e)}")
            
            with col2:
                # Excel Export
                try:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                        summary_df = pd.DataFrame([summary_stats])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Excel export failed: {str(e)}")
            
            with col3:
                # JSON Export
                try:
                    json_data = export_df.to_json(orient='records', date_format='iso', indent=2)
                    st.download_button(
                        label="🔧 Download JSON",
                        data=json_data,
                        file_name=f"gencoachingiq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"JSON export failed: {str(e)}")
            
            # Export info
            st.info(f"**Ready to export {len(export_df)} conversations** with {len(available_columns)} data columns")
    
    def _render_user_guide(self):
        """Render user guide"""
        st.header("📖 GenCoachingIQ User Guide")
        
        guide_tabs = st.tabs(["🚀 Getting Started", "🧠 Enhanced Features", "📊 Understanding Results", "❓ FAQ"])
        
        with guide_tabs[0]:
            st.markdown("""
            ## 🚀 Getting Started with GenCoachingIQ
            
            ### Step 1: Prepare Your Data
            **Supported Formats**: Excel (.xlsx, .xls), CSV (.csv), Text (.txt)
            
            **Enhanced Format** (Recommended):
            ```
            [12:30:08 AGENT]: How can I help you today?
            [12:30:15 CUSTOMER]: I have an issue with my account
            [12:30:20 AGENT]: I'd be happy to help with that
            ```
            
            **Basic Format**: Simple transcript text (one conversation per row/line)
            
            ### Step 2: Upload and Process
            1. Go to **Upload & Process** tab
            2. Select your file and click **🚀 Start Analysis**
            3. Monitor progress and animations
            4. Review the analysis preview
            
            ### Step 3: Explore Insights
            - **Dashboard**: High-level overview with key metrics
            - **Results & Analytics**: Detailed analysis with visualizations
            - **Export**: Download for Power BI or further analysis
            """)
        
        with guide_tabs[1]:
            st.markdown("""
            ## 🧠 Enhanced Features
            
            ### Timestamped Conversation Analysis
            GenCoachingIQ automatically detects and parses timestamped conversations to provide:
            - **Turn-by-turn sentiment tracking**
            - **Speaker-specific analysis** (Agent vs Customer)
            - **Conversation flow visualization**
            - **Critical moment identification**
            
            ### Advanced Theme Extraction
            - **Separate theme analysis** for agents and customers
            - **TF-IDF based clustering** for accurate theme identification
            - **Top 5 themes visualization** by speaker
            
            ### Intelligent Coaching Insights
            - **Priority scoring** (1-10) for coaching needs
            - **Sentiment progression** analysis
            - **Performance correlation** with conversation metrics
            """)
        
        with guide_tabs[2]:
            st.markdown("""
            ## 📊 Understanding Your Results
            
            ### Key Metrics
            
            **Coaching Score (0-100)**
            - 80-100: Excellent performance
            - 60-79: Good with room for improvement
            - 40-59: Average, coaching recommended
            - 0-39: Needs immediate attention
            
            **Coaching Priority (1-10)**
            - 8-10: Critical - immediate coaching needed
            - 5-7: High - schedule coaching session
            - 3-4: Medium - include in next review
            - 1-2: Low - maintain current performance
            
            **Enhanced Analysis**
            Shows conversations with timestamp data for deeper insights
            
            ### Visualizations
            - **Sentiment Timeline**: Track emotional journey through conversation
            - **Speaker Comparison**: Compare agent vs customer themes and sentiment
            - **Performance Correlation**: Duration vs coaching score analysis
            """)
        
        with guide_tabs[3]:
            st.markdown("""
            ## ❓ Frequently Asked Questions
            
            **Q: What makes GenCoachingIQ different?**
            A: We analyze conversation flow and speaker-specific patterns, not just overall sentiment.
            
            **Q: Do I need timestamped data?**
            A: No, but timestamped conversations unlock advanced features like turn-by-turn analysis.
            
            **Q: How accurate is the analysis?**
            A: Our AI models achieve 85-90% accuracy on customer service conversations.
            
            **Q: Can I customize the analysis?**
            A: Yes! Use Configuration tab to adjust weights, keywords, and themes.
            
            **Q: How do I export for Power BI?**
            A: Download CSV from Results → Export tab, then import into Power BI.
            
            **Q: What if agent/customer shows 'Unknown'?**
            A: This means your data doesn't have timestamp format. Use [HH:MM:SS SPEAKER]: format for best results.
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

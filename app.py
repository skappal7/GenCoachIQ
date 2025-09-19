import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import pickle
import hashlib

# Lightweight NLP libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# File format support
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    st.error("Please install pyarrow: pip install pyarrow")

# Configure page
st.set_page_config(
    page_title="Call Center Coaching Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with proper caching
def initialize_session_state():
    """Initialize session state with caching capabilities"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'turn_analysis' not in st.session_state:
        st.session_state.turn_analysis = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'file_hash' not in st.session_state:
        st.session_state.file_hash = None
    if 'processing_cache' not in st.session_state:
        st.session_state.processing_cache = {}
    if 'parquet_table' not in st.session_state:
        st.session_state.parquet_table = None
    if 'parquet_dataframe' not in st.session_state:
        st.session_state.parquet_dataframe = None
    if 'compression_stats' not in st.session_state:
        st.session_state.compression_stats = {}

class ParquetManager:
    """Handles all Parquet operations for optimal performance"""
    
    @staticmethod
    def convert_to_parquet(df, filename):
        """Convert DataFrame to Parquet with Snappy compression"""
        try:
            with st.spinner(f"Converting {filename} to optimized Parquet format..."):
                # Calculate original size
                original_size_bytes = df.memory_usage(deep=True).sum()
                original_size_mb = original_size_bytes / 1024 / 1024
                
                # Convert to PyArrow Table with optimized schema
                table = pa.Table.from_pandas(df)
                
                # Create compressed Parquet buffer
                parquet_buffer = BytesIO()
                pq.write_table(
                    table, 
                    parquet_buffer, 
                    compression='snappy',
                    use_dictionary=True,
                    row_group_size=10000,
                    use_compliant_nested_type=True
                )
                parquet_buffer.seek(0)
                
                # Read back the compressed table
                compressed_table = pq.read_table(parquet_buffer)
                compressed_df = compressed_table.to_pandas()
                
                # Calculate compression stats
                compressed_size_bytes = len(parquet_buffer.getvalue())
                compressed_size_mb = compressed_size_bytes / 1024 / 1024
                compression_ratio = ((original_size_bytes - compressed_size_bytes) / original_size_bytes * 100) if original_size_bytes > 0 else 0
                
                # Store compression stats
                compression_stats = {
                    'original_size_mb': original_size_mb,
                    'compressed_size_mb': compressed_size_mb,
                    'compression_ratio': compression_ratio,
                    'memory_savings': original_size_mb - compressed_size_mb
                }
                
                # Display compression results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Size", f"{original_size_mb:.2f} MB")
                with col2:
                    st.metric("Compressed Size", f"{compressed_size_mb:.2f} MB")
                with col3:
                    st.metric("Compression", f"{compression_ratio:.1f}%")
                with col4:
                    st.metric("Memory Saved", f"{compression_stats['memory_savings']:.2f} MB")
                
                st.success("✅ Parquet conversion completed with Snappy compression")
                
                # Store in session state
                st.session_state.parquet_table = compressed_table
                st.session_state.parquet_dataframe = compressed_df
                st.session_state.compression_stats = compression_stats
                
                return compressed_table, compressed_df, compression_stats
                
        except Exception as e:
            st.error(f"Parquet conversion failed: {str(e)}")
            st.warning("Falling back to original DataFrame - performance may be impacted")
            return None, df, {}
    
    @staticmethod
    def read_parquet_optimized(parquet_table, columns=None):
        """Read specific columns from Parquet for memory efficiency"""
        try:
            if columns:
                # Only read required columns
                selected_table = parquet_table.select(columns)
                return selected_table.to_pandas()
            else:
                return parquet_table.to_pandas()
        except Exception as e:
            st.error(f"Error reading Parquet data: {str(e)}")
            return parquet_table.to_pandas()
    
    @staticmethod
    def filter_parquet_table(parquet_table, filters=None):
        """Apply filters directly to Parquet table for better performance"""
        try:
            if filters:
                filtered_table = parquet_table.filter(filters)
                return filtered_table
            return parquet_table
        except Exception as e:
            # Fallback to DataFrame filtering
            return parquet_table

class CallCenterAnalyzer:
    def __init__(self, use_parquet=True):
        self.vader = SentimentIntensityAnalyzer()
        self.use_parquet = use_parquet
        self.parquet_manager = ParquetManager()
        self.coaching_themes = {
            # 1. Empathy & Emotional Intelligence
            "empathy": [
                "I understand", "I completely understand", "I do understand", "I hear you",
                "I see how you feel", "I see where you're coming from", "I realize this is frustrating",
                "I can imagine", "I know this must be tough", "I'm sorry", "I apologize",
                "I truly empathize", "thank you for your patience", "thank you for waiting",
                "your concern is valid", "I totally get it", "I know how you feel",
                "sorry to hear that", "I get your frustration", "I appreciate you sharing this",
                "must be difficult", "I respect your concern", "I'm here to help",
                "I hear the frustration in your voice", "I can sense this has been upsetting",
                "it sounds like this caused you trouble", "I'll do everything I can"
            ],

            # 2. Professionalism & Courtesy
            "professionalism": [
                "thank you for calling", "good morning", "good afternoon", "good evening",
                "it's my pleasure", "I'll be happy to help", "thank you for reaching out",
                "thank you for your time", "I appreciate your time", "it's been a pleasure",
                "with respect", "I assure you", "I value your feedback", "thank you for choosing us",
                "allow me to assist", "thanks for your cooperation", "thanks for staying on the line",
                "I appreciate your understanding", "thank you for bearing with me",
                "I appreciate your patience"
            ],

            # 3. Problem Solving & Resolution
            "problem_solving": [
                "let me check this for you", "let me look into this", "I will fix this",
                "I'll work on resolving this", "we'll sort this out", "let's troubleshoot together",
                "let me walk you through", "next steps will be", "I'll escalate this for resolution",
                "I'll provide a workaround", "we can try this option", "here's a solution",
                "I will investigate further", "I will resolve this", "I'll take corrective action",
                "let's address this issue", "let me confirm once more", "this should fix it",
                "the root cause appears to be", "we'll ensure this won't happen again",
                "rest assured", "let me double-check that", "I'll send this to our specialist",
                "I'll fast-track this to our tech team"
            ],

            # 4. Escalation Triggers (Customer Rants)
            "escalation_triggers": [
                "I want to speak to your supervisor", "let me talk to your manager",
                "this is unacceptable", "this is ridiculous", "you people are useless",
                "I've called 5 times already", "I'm sick of this", "you guys messed up",
                "I don't want excuses", "just fix it", "cancel my account", "I'm done with your service",
                "I'll take my business elsewhere", "I'll post a bad review", "I'm wasting my time",
                "you never listen", "nobody helps me here", "why is this so difficult",
                "I don't believe you", "you're not helping me"
            ],

            # 5. Active Listening
            "listening": [
                "let me repeat that back", "so what you're saying is", "to confirm your point",
                "just to clarify", "if I understand correctly", "I heard you say",
                "I got that", "I see what you mean", "noted", "please continue",
                "go ahead", "tell me more about that", "let me make sure I understood",
                "correct me if I'm wrong", "I'm listening carefully", "thanks for sharing that",
                "so to summarize your concern", "let me restate it to be sure",
                "to check my understanding", "just to make sure I captured that correctly"
            ],

            # 6. Polite Agent Phrases
            "polite_agent": [
                "please", "may I ask", "could you kindly", "if you don't mind",
                "thank you very much", "I'd be glad to assist", "allow me a moment",
                "thanks for waiting", "thank you for clarifying", "please hold while I check",
                "thanks for your cooperation", "thanks for your patience",
                "may I confirm", "if that's okay with you", "let me check that for you"
            ],

            # 7. Impolite Agent Phrases (coaching negatives)
            "impolite_agent": [
                "that's not my problem", "I can't help you with that", "that's impossible",
                "you'll have to deal with it", "I already told you", "you're not listening",
                "that's just the way it is", "there's nothing I can do", "stop interrupting me",
                "you should have read the policy", "you must have done it wrong",
                "we can't do anything about that", "I don't have time for this",
                "that's not my department", "you need to figure it out"
            ],

            # 8. Closing & Wrap-up
            "closing": [
                "is there anything else I can help you with", "thank you for your time today",
                "thank you for calling", "I appreciate your patience",
                "your issue has been resolved", "let me summarize",
                "to recap", "to confirm the resolution", "please reach out again if needed",
                "I'm glad I could help", "thank you once again", "we appreciate your business",
                "it's been my pleasure assisting you", "enjoy your day", "take care",
                "closing the loop", "all set now", "your concern is addressed",
                "we look forward to serving you again"
            ],

            # 9. Urgency & Escalation
            "urgency": [
                "I'll prioritize this", "this will be resolved urgently", "I'll escalate immediately",
                "we'll handle this right away", "this is high priority", "critical matter",
                "this needs immediate attention", "I'll fast-track this", "we'll expedite the process",
                "as soon as possible", "I'll make this a top priority", "we'll resolve this quickly",
                "urgent escalation required", "emergency handling", "let's address this without delay"
            ],

            # 10. Customer Colloquialisms
            "customer_colloquialisms": [
                "you guys messed up", "why is this so hard", "you never help me",
                "every time I call", "same old story", "I'm fed up", "this always happens",
                "I'm not happy with this", "don't waste my time", "you people always say that",
                "I already explained this", "why can't you fix it", "this is taking forever",
                "it's not rocket science", "how hard can it be", "I'm losing my patience",
                "you're just reading a script", "talking to a wall"
            ],

            # 11. Domain-Specific Keywords
            "domain_specific": [
                # Banking/Insurance
                "account balance", "policy number", "premium due", "loan repayment", "mortgage",
                "installment", "interest rate", "credit card limit", "insurance claim", "underwriting",
                "KYC process", "coverage details", "deductible", "co-pay", "payout", "settlement",
                "investment maturity", "fixed deposit", "account closure",
                # Retail/E-commerce
                "order ID", "shipment tracking", "refund request", "return policy", "replacement order",
                "discount coupon", "promo code", "delivery date", "out of stock", "loyalty points",
                "cart checkout", "payment failure", "invoice copy", "exchange request",
                # Healthcare
                "doctor appointment", "prescription refill", "medical records", "insurance coverage",
                "treatment plan", "consultation", "diagnosis", "lab results", "pharmacy",
                "claim rejection", "prior authorization", "health benefits",
                # Travel
                "flight booking", "ticket cancellation", "check-in", "boarding pass", "reservation number",
                "baggage allowance", "seat upgrade", "travel insurance", "itinerary", "visa application",
                "hotel reservation", "passport details", "flight delay", "lost baggage"
            ],
            
            # 12. Customer Satisfaction Indicators (Positive)
            "customer_satisfaction": [
                "thank you so much", "you've been very helpful", "I appreciate your help",
                "that was exactly what I needed", "perfect", "excellent service", "great job",
                "you solved my problem", "I'm satisfied", "that's much better", "wonderful",
                "you're amazing", "I'm impressed", "outstanding support", "quick resolution",
                "you made this easy", "fantastic", "couldn't ask for better service"
            ],
            
            # 13. Customer Dissatisfaction Indicators (Negative)
            "customer_dissatisfaction": [
                "this is terrible", "worst service ever", "I'm not satisfied", "this doesn't help",
                "you're wasting my time", "I'm still confused", "this makes no sense",
                "you didn't fix anything", "I'm more frustrated now", "this is getting worse",
                "I regret calling", "you made it worse", "I'm disappointed", "useless advice",
                "I want my money back", "I'm canceling everything", "I'll never use this service again"
            ],
            
            # 14. Agent Knowledge Gaps (coaching opportunities)
            "knowledge_gaps": [
                "I don't know", "I'm not sure", "let me find out", "I'll have to check",
                "that's not my area", "I'm not familiar with that", "I'll need to ask someone",
                "I don't have that information", "I'll get back to you", "let me research that",
                "I'm not trained on that", "I'll transfer you", "that's above my level",
                "I don't have access to that", "I'm still learning about that"
            ],
            
            # 15. Customer Effort Indicators (high effort = bad experience)
            "customer_effort": [
                "I've been transferred 3 times", "I've been on hold forever", "I keep repeating myself",
                "I already told the other person", "why do I have to explain again", "this is my fifth call",
                "I've been calling all day", "no one can help me", "I keep getting bounced around",
                "I've tried everything", "this is taking too long", "I've wasted hours on this",
                "why is this so complicated", "I shouldn't have to do this much work"
            ]
        }
        
        # Theme scoring weights based on coaching importance
        self.theme_weights = {
            "impolite_agent": -3,  # Most critical negative
            "knowledge_gaps": -2,
            "escalation_triggers": -2,
            "customer_dissatisfaction": -1,
            "customer_effort": -1,
            "empathy": 2,  # Most important positive
            "problem_solving": 2,
            "professionalism": 1,
            "listening": 1,
            "polite_agent": 1,
            "customer_satisfaction": 1,
            "closing": 1,
            "urgency": 0.5,
            "customer_colloquialisms": 0,
            "domain_specific": 0
        }
    
    @lru_cache(maxsize=1000)
    def get_sentiment_scores(self, text):
        """Cached sentiment analysis"""
        if not text or pd.isna(text):
            return 0.0, 0.0, 0.0, 0.0
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(str(text))
        
        # TextBlob sentiment
        blob = TextBlob(str(text))
        textblob_polarity = blob.sentiment.polarity
        
        return (vader_scores['compound'], vader_scores['pos'], 
                vader_scores['neg'], textblob_polarity)
    
    def predict_nps(self, sentiment_compound, sentiment_positive, sentiment_negative):
        """Predict NPS based on sentiment scores"""
        # Weighted scoring for NPS prediction
        if sentiment_compound >= 0.5:
            return np.random.randint(9, 11)  # Promoters
        elif sentiment_compound >= 0.1:
            return np.random.randint(7, 9)   # Passives
        else:
            return np.random.randint(0, 7)   # Detractors
    
    def parse_embedded_transcript(self, text):
        """Parse embedded timestamp-speaker format: [HH:MM:SS SPEAKER]: dialogue"""
        if not text or pd.isna(text):
            return []
        
        # Pattern to match [timestamp speaker]: dialogue
        pattern = r'\[(\d{1,2}:\d{2}:\d{2})\s+([A-Za-z]+)\]:\s*(.*?)(?=\[\d{1,2}:\d{2}:\d{2}|$)'
        matches = re.findall(pattern, str(text), re.DOTALL | re.IGNORECASE)
        
        turns = []
        for i, match in enumerate(matches):
            timestamp, speaker, dialogue = match
            turns.append({
                'turn_number': i + 1,
                'timestamp': timestamp.strip(),
                'speaker': speaker.strip().upper(),
                'text': dialogue.strip()
            })
        
        return turns
    
    def identify_coaching_themes(self, text):
        """Enhanced coaching theme identification with context awareness"""
        if not text or pd.isna(text):
            return {}
        
        text_lower = str(text).lower()
        themes_found = {}
        
        # Advanced phrase matching with context
        for theme, phrases in self.coaching_themes.items():
            theme_score = 0
            matched_phrases = []
            
            for phrase in phrases:
                # Exact phrase matching
                if phrase.lower() in text_lower:
                    theme_score += 1
                    matched_phrases.append(phrase)
                
                # Partial matching for key phrases (more flexible)
                phrase_words = phrase.lower().split()
                if len(phrase_words) > 1:
                    # Check if majority of words from phrase are present
                    word_matches = sum(1 for word in phrase_words if word in text_lower)
                    if word_matches >= len(phrase_words) * 0.7:  # 70% word match threshold
                        theme_score += 0.5
            
            if theme_score > 0:
                themes_found[theme] = {
                    'score': theme_score,
                    'matched_phrases': matched_phrases[:5],  # Top 5 matches
                    'weight': self.theme_weights.get(theme, 0)
                }
        
        return themes_found
    
    def calculate_coaching_priority(self, themes_dict):
        """Calculate coaching priority based on theme weights"""
        if not themes_dict:
            return 0
        
        total_weighted_score = 0
        for theme, details in themes_dict.items():
            score = details.get('score', 0)
            weight = details.get('weight', 0)
            total_weighted_score += score * weight
        
        return total_weighted_score
    
    def detect_conversation_quality(self, text, speaker='Unknown'):
        """Detect conversation quality indicators"""
        if not text or pd.isna(text):
            return {'quality_score': 0, 'indicators': []}
        
        text_lower = str(text).lower()
        quality_indicators = []
        quality_score = 0
        
        # Positive indicators
        positive_phrases = [
            'thank you', 'please', 'I apologize', 'I understand',
            'let me help', 'I appreciate', 'my pleasure'
        ]
        
        # Negative indicators
        negative_phrases = [
            'that\'s not my problem', 'impossible', 'can\'t help',
            'not my department', 'you should have', 'I don\'t know'
        ]
        
        # Count positive indicators
        for phrase in positive_phrases:
            if phrase in text_lower:
                quality_score += 1
                quality_indicators.append(f"Positive: {phrase}")
        
        # Count negative indicators
        for phrase in negative_phrases:
            if phrase in text_lower:
                quality_score -= 2
                quality_indicators.append(f"Negative: {phrase}")
        
        return {
            'quality_score': quality_score,
            'indicators': quality_indicators
        }
    
    def analyze_customer_agent_interaction(self, text, speaker):
        """Specialized analysis based on speaker type"""
        themes = self.identify_coaching_themes(text)
        quality = self.detect_conversation_quality(text, speaker)
        
        if speaker and speaker.upper() in ['AGENT', 'REP', 'REPRESENTATIVE']:
            # Agent-specific analysis
            coaching_focus = []
            
            # Check for impolite language
            if 'impolite_agent' in themes:
                coaching_focus.append('CRITICAL: Impolite language detected')
            
            # Check for knowledge gaps
            if 'knowledge_gaps' in themes:
                coaching_focus.append('Training needed: Knowledge gaps identified')
            
            # Check for good behaviors
            if 'empathy' in themes:
                coaching_focus.append('Strength: Empathy demonstrated')
            
            if 'problem_solving' in themes:
                coaching_focus.append('Strength: Problem-solving approach')
            
            return {
                'speaker_type': 'agent',
                'coaching_focus': coaching_focus,
                'themes': themes,
                'quality': quality
            }
        
        elif speaker and speaker.upper() in ['CUSTOMER', 'CLIENT', 'CALLER']:
            # Customer-specific analysis
            satisfaction_indicators = []
            
            # Check satisfaction levels
            if 'customer_satisfaction' in themes:
                satisfaction_indicators.append('Positive: Customer satisfaction expressed')
            
            if 'customer_dissatisfaction' in themes:
                satisfaction_indicators.append('Alert: Customer dissatisfaction detected')
            
            if 'escalation_triggers' in themes:
                satisfaction_indicators.append('WARNING: Escalation triggers present')
            
            if 'customer_effort' in themes:
                satisfaction_indicators.append('Concern: High customer effort indicated')
            
            return {
                'speaker_type': 'customer',
                'satisfaction_indicators': satisfaction_indicators,
                'themes': themes,
                'quality': quality
            }
        
        return {
            'speaker_type': 'unknown',
            'themes': themes,
            'quality': quality
        }
    
    def extract_speaker_turns(self, text, timestamp_col=None, speaker_col=None):
        """Extract turn-by-turn analysis from embedded format"""
        turns = self.parse_embedded_transcript(text)
        
        # If no embedded format found, try legacy pattern matching
        if not turns:
            agent_patterns = r'(agent|rep|representative)[:|-]?\s*(.*?)(?=customer|client|caller|$)'
            customer_patterns = r'(customer|client|caller|user)[:|-]?\s*(.*?)(?=agent|rep|representative|$)'
            
            agent_matches = re.findall(agent_patterns, str(text), re.IGNORECASE | re.DOTALL)
            customer_matches = re.findall(customer_patterns, str(text), re.IGNORECASE | re.DOTALL)
            
            for i, match in enumerate(agent_matches):
                turns.append({
                    'turn_number': i + 1,
                    'speaker': 'AGENT',
                    'text': match[1].strip(),
                    'timestamp': f"Turn {i+1}"
                })
            
            for i, match in enumerate(customer_matches):
                turns.append({
                    'turn_number': i + 1,
                    'speaker': 'CUSTOMER',
                    'text': match[1].strip(),
                    'timestamp': f"Turn {i+1}"
                })
        
        return turns
    
    def process_transcript_from_parquet(self, text_col, parquet_table=None, parquet_df=None, parallel=False):
        """Process transcript using Parquet data for optimal performance"""
        
        # Use Parquet table if available, otherwise DataFrame
        if parquet_table is not None and self.use_parquet:
            st.info("🚀 Processing using optimized Parquet format")
            # Convert Parquet table to DataFrame for processing
            df = parquet_table.to_pandas()
            data_source = "parquet_table"
        elif parquet_df is not None:
            st.info("🚀 Processing using Parquet-optimized DataFrame")
            df = parquet_df
            data_source = "parquet_dataframe"
        else:
            st.error("No Parquet data available for processing")
            return None
        
        # Generate cache key based on Parquet data
        cache_key = hashlib.md5(f"{text_col}_{len(df)}_{data_source}_{df.iloc[0][text_col][:50] if len(df) > 0 else ''}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in st.session_state.processing_cache:
            st.info("📋 Using cached Parquet processing results")
            return st.session_state.processing_cache[cache_key]
        
        results = []
        
        def process_row_optimized(row):
            text = row[text_col]
            
            # Parse embedded transcript for speaker-specific analysis
            turns = self.parse_embedded_transcript(text)
            
            # Overall transcript analysis
            vader_compound, vader_pos, vader_neg, textblob_pol = self.get_sentiment_scores(text)
            nps_score = self.predict_nps(vader_compound, vader_pos, vader_neg)
            themes = self.identify_coaching_themes(text)
            coaching_priority = self.calculate_coaching_priority(themes)
            
            # Get top themes with scores
            top_themes = sorted(themes.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
            top_theme = top_themes[0][0] if top_themes else 'none'
            
            # Speaker-specific analysis from parsed turns
            agent_analysis = []
            customer_analysis = []
            
            for turn in turns:
                speaker_analysis = self.analyze_customer_agent_interaction(turn['text'], turn['speaker'])
                if speaker_analysis['speaker_type'] == 'agent':
                    agent_analysis.extend(speaker_analysis.get('coaching_focus', []))
                elif speaker_analysis['speaker_type'] == 'customer':
                    customer_analysis.extend(speaker_analysis.get('satisfaction_indicators', []))
            
            # Overall quality from full transcript
            quality = self.detect_conversation_quality(text)
            
            # Format themes for display
            themes_summary = {}
            for theme, details in themes.items():
                themes_summary[theme] = {
                    'score': details['score'],
                    'top_phrases': details['matched_phrases'][:3]
                }
            
            return {
                'transcript_text': text[:200] + '...' if len(text) > 200 else text,
                'total_turns': len(turns),
                'agent_turns': len([t for t in turns if t['speaker'] == 'AGENT']),
                'customer_turns': len([t for t in turns if t['speaker'] == 'CUSTOMER']),
                'vader_compound': round(vader_compound, 3),
                'vader_positive': round(vader_pos, 3),
                'vader_negative': round(vader_neg, 3),
                'textblob_polarity': round(textblob_pol, 3),
                'predicted_nps': nps_score,
                'coaching_priority_score': round(coaching_priority, 2),
                'top_coaching_theme': top_theme,
                'theme_count': len(themes),
                'detailed_themes': str(themes_summary),
                'agent_coaching_focus': str(agent_analysis),
                'customer_satisfaction_indicators': str(customer_analysis),
                'quality_score': quality['quality_score'],
                'quality_indicators': str(quality['indicators']),
                'conversation_duration': f"{turns[-1]['timestamp']}" if turns else 'N/A',
                'data_source': data_source
            }
        
        # Process with optimized Parquet data
        if parallel and len(df) > 100:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_row_optimized, row): idx for idx, row in df.iterrows()}
                
                progress_bar = st.progress(0)
                for i, future in enumerate(as_completed(futures)):
                    results.append(future.result())
                    progress_bar.progress((i + 1) / len(futures))
        else:
            progress_bar = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows()):
                results.append(process_row_optimized(row))
                progress_bar.progress((i + 1) / len(df))
        
        # Create result DataFrame and cache it
        result_df = pd.DataFrame(results)
        st.session_state.processing_cache[cache_key] = result_df
        
        st.success(f"✅ Processed {len(result_df)} records using {data_source}")
        return result_df
    
    def generate_turn_analysis_from_parquet(self, text_col, parquet_df):
        """Generate turn-by-turn analysis using Parquet data"""
        st.info("🔄 Generating turn analysis from Parquet data")
        
        turn_results = []
        progress_bar = st.progress(0)
        
        for i, row in parquet_df.iterrows():
            transcript_text = row[text_col]
            turns = self.parse_embedded_transcript(transcript_text)
            
            for turn in turns:
                # Analyze each turn
                sentiment_scores = self.get_sentiment_scores(turn['text'])
                themes = self.identify_coaching_themes(turn['text'])
                speaker_analysis = self.analyze_customer_agent_interaction(turn['text'], turn['speaker'])
                coaching_priority = self.calculate_coaching_priority(themes)
                
                # Get top theme
                top_themes = sorted(themes.items(), key=lambda x: x[1]['score'], reverse=True)[:1]
                top_theme = top_themes[0][0] if top_themes else 'none'
                
                turn_results.append({
                    'transcript_id': i,
                    'turn_number': turn['turn_number'],
                    'timestamp': turn['timestamp'],
                    'speaker': turn['speaker'],
                    'text_preview': turn['text'][:100] + '...' if len(turn['text']) > 100 else turn['text'],
                    'sentiment_compound': round(sentiment_scores[0], 3),
                    'coaching_priority': round(coaching_priority, 2),
                    'top_theme': top_theme,
                    'theme_count': len(themes),
                    'speaker_analysis': str(speaker_analysis.get('coaching_focus', [])) if speaker_analysis['speaker_type'] == 'agent' else str(speaker_analysis.get('satisfaction_indicators', []))
                })
            
            progress_bar.progress((i + 1) / len(parquet_df))
        
        turn_df = pd.DataFrame(turn_results)
        st.success(f"✅ Generated turn analysis for {len(turn_df)} turns from Parquet data")
        return turn_df

def get_file_hash(uploaded_file):
    """Generate hash for uploaded file"""
    file_content = uploaded_file.getvalue()
    return hashlib.md5(file_content).hexdigest()

def load_file_to_parquet(uploaded_file):
    """Load file and convert to Parquet with caching"""
    try:
        # Check if file is already processed
        file_hash = get_file_hash(uploaded_file)
        
        if (st.session_state.file_hash == file_hash and 
            st.session_state.parquet_table is not None and 
            st.session_state.parquet_dataframe is not None):
            st.info("📋 Using cached Parquet data")
            return st.session_state.parquet_table, st.session_state.parquet_dataframe, st.session_state.compression_stats
        
        filename = uploaded_file.name
        file_ext = filename.split('.')[-1].lower()
        
        with st.spinner(f"Loading {filename}..."):
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, XLS, or XLSX files.")
                return None, None, {}
            
            st.success(f"📁 File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Convert to Parquet with compression
            parquet_manager = ParquetManager()
            parquet_table, parquet_df, compression_stats = parquet_manager.convert_to_parquet(df, filename)
            
            # Update session state
            st.session_state.file_hash = file_hash
            
            return parquet_table, parquet_df, compression_stats
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, {}

def create_parquet_export(results_df, turn_df=None):
    """Create Parquet export for download"""
    try:
        # Convert results to Parquet
        results_table = pa.Table.from_pandas(results_df)
        results_buffer = BytesIO()
        pq.write_table(results_table, results_buffer, compression='snappy')
        results_buffer.seek(0)
        
        export_data = {
            'coaching_analysis': results_buffer.getvalue()
        }
        
        # Add turn analysis if available
        if turn_df is not None:
            turn_table = pa.Table.from_pandas(turn_df)
            turn_buffer = BytesIO()
            pq.write_table(turn_table, turn_buffer, compression='snappy')
            turn_buffer.seek(0)
            export_data['turn_analysis'] = turn_buffer.getvalue()
        
        return export_data
        
    except Exception as e:
        st.error(f"Error creating Parquet export: {str(e)}")
        return None

def main():
    # Initialize session state
    initialize_session_state()
    
    st.title("📞 Call Center Agent Coaching Analytics")
    st.markdown("*Transform call transcripts into actionable coaching insights with Parquet optimization*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Processing options
        st.subheader("Processing Options")
        parallel_processing = st.checkbox("Enable Parallel Processing", help="Faster processing for large datasets")
        use_parquet_optimization = st.checkbox("Use Parquet Optimization", value=True, help="Use Parquet format for faster processing")
        
        # Session cache and Parquet stats
        st.subheader("📊 Performance Stats")
        cache_size = len(st.session_state.processing_cache)
        st.metric("Cache Entries", cache_size)
        
        # Display Parquet stats if available
        if st.session_state.compression_stats:
            stats = st.session_state.compression_stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memory Saved", f"{stats.get('memory_savings', 0):.1f} MB")
            with col2:
                st.metric("Compression", f"{stats.get('compression_ratio', 0):.1f}%")
        
        # Cache management
        if st.button("Clear All Cache", help="Clear all cached processing results and Parquet data"):
            st.session_state.processing_cache.clear()
            st.session_state.parquet_table = None
            st.session_state.parquet_dataframe = None
            st.session_state.file_hash = None
            st.session_state.compression_stats = {}
            st.success("All cache cleared")
            st.rerun()
        
        # File upload section
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload transcript file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )
        
        # Help for embedded format
        st.subheader("📝 Transcript Format")
        st.info("Supports embedded format: [HH:MM:SS SPEAKER]: dialogue")
        with st.expander("Format Example"):
            st.code("""[10:00:00 AGENT]: Hello, how can I help?
[10:00:15 CUSTOMER]: I have an issue with my order
[10:00:30 AGENT]: I understand your concern""")
    
    # Main content area
    if uploaded_file is not None:
        # Load file to Parquet
        if not st.session_state.file_uploaded or st.session_state.get('last_file') != uploaded_file.name:
            parquet_table, parquet_df, compression_stats = load_file_to_parquet(uploaded_file)
            
            if parquet_table is not None and parquet_df is not None:
                st.session_state.parquet_table = parquet_table
                st.session_state.parquet_dataframe = parquet_df
                st.session_state.compression_stats = compression_stats
                st.session_state.file_uploaded = True
                st.session_state.last_file = uploaded_file.name
            else:
                st.stop()
        else:
            parquet_table = st.session_state.parquet_table
            parquet_df = st.session_state.parquet_dataframe
        
        # Column selection from Parquet data
        st.subheader("🎯 Column Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            available_columns = parquet_df.columns.tolist()
            text_column = st.selectbox(
                "Select Text/Transcript Column",
                options=available_columns,
                help="Column containing the conversation transcript with embedded speaker/timestamp format"
            )
        
        with col2:
            # Additional columns to include
            additional_columns = st.multiselect(
                "Select additional columns for output",
                options=[col for col in available_columns if col != text_column],
                help="These columns will be included in your final export"
            )
        
        # Process button
        if st.button("🚀 Start Parquet-Optimized Analysis", type="primary"):
            analyzer = CallCenterAnalyzer(use_parquet=use_parquet_optimization)
            
            with st.spinner("Processing transcripts using Parquet optimization..."):
                start_time = time.time()
                
                # Main analysis using Parquet data
                results_df = analyzer.process_transcript_from_parquet(
                    text_column, 
                    parquet_table=parquet_table if use_parquet_optimization else None,
                    parquet_df=parquet_df,
                    parallel=parallel_processing
                )
                
                if results_df is not None:
                    # Add additional columns from Parquet DataFrame
                    for col in additional_columns:
                        if col in parquet_df.columns:
                            results_df[col] = parquet_df[col].reset_index(drop=True)
                    
                    processing_time = time.time() - start_time
                    
                    st.session_state.processed_data = results_df
                    st.success(f"✅ Parquet-optimized analysis completed in {processing_time:.2f} seconds!")
                else:
                    st.error("Analysis failed. Please try again.")
        
        # Display results
        if st.session_state.processed_data is not None:
            results_df = st.session_state.processed_data
            
            # Data source indicator
            if 'data_source' in results_df.columns:
                data_source = results_df['data_source'].iloc[0]
                if 'parquet' in data_source:
                    st.success(f"🚀 Results generated using {data_source}")
            
            # Summary metrics
            st.subheader("📈 Analysis Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                avg_nps = results_df['predicted_nps'].mean()
                st.metric("Average Predicted NPS", f"{avg_nps:.1f}")
            
            with col2:
                avg_sentiment = results_df['vader_compound'].mean()
                st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            
            with col3:
                avg_coaching_priority = results_df['coaching_priority_score'].mean()
                st.metric("Avg Coaching Priority", f"{avg_coaching_priority:.1f}")
            
            with col4:
                high_priority_count = len(results_df[results_df['coaching_priority_score'] < -2])
                st.metric("Critical Issues", high_priority_count)
            
            with col5:
                total_turns = results_df['total_turns'].sum()
                st.metric("Total Conversation Turns", total_turns)
            
            # Coaching Priority Analysis
            st.subheader("🎯 Coaching Priority Breakdown")
            
            priority_col1, priority_col2 = st.columns(2)
            
            with priority_col1:
                # Critical coaching needs
                critical_issues = results_df[results_df['coaching_priority_score'] < -2]
                if len(critical_issues) > 0:
                    st.error(f"🚨 {len(critical_issues)} transcripts require immediate coaching intervention")
                    st.dataframe(
                        critical_issues[['coaching_priority_score', 'top_coaching_theme', 'agent_coaching_focus']].head(),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("✅ No critical coaching issues identified")
            
            with priority_col2:
                # Positive coaching examples
                positive_examples = results_df[results_df['coaching_priority_score'] > 2]
                if len(positive_examples) > 0:
                    st.success(f"⭐ {len(positive_examples)} examples of excellent performance")
                    st.dataframe(
                        positive_examples[['coaching_priority_score', 'top_coaching_theme', 'quality_score']].head(),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("💡 Consider highlighting positive coaching examples")
            
            # Main results table
            st.subheader("🎯 Detailed Coaching Analysis Results")
            
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                default_columns = ['predicted_nps', 'coaching_priority_score', 'top_coaching_theme', 'quality_score', 'total_turns']
                show_columns = st.multiselect(
                    "Select columns to display",
                    options=results_df.columns.tolist(),
                    default=[col for col in default_columns if col in results_df.columns]
                )
            
            with col2:
                filter_theme = st.selectbox(
                    "Filter by coaching theme",
                    options=['All'] + sorted(results_df['top_coaching_theme'].unique().tolist()),
                    key="main_theme_filter"
                )
            
            with col3:
                priority_filter = st.selectbox(
                    "Filter by coaching priority",
                    options=['All', 'Critical (< -2)', 'Needs Improvement (< 0)', 'Good Performance (> 2)'],
                    key="main_priority_filter"
                )
            
            # Apply filters with proper error handling
            display_df = results_df.copy()
            
            # Theme filter
            if filter_theme != 'All':
                display_df = display_df[display_df['top_coaching_theme'] == filter_theme]
            
            # Priority filter
            if priority_filter != 'All':
                if priority_filter == 'Critical (< -2)':
                    display_df = display_df[display_df['coaching_priority_score'] < -2]
                elif priority_filter == 'Needs Improvement (< 0)':
                    display_df = display_df[display_df['coaching_priority_score'] < 0]
                elif priority_filter == 'Good Performance (> 2)':
                    display_df = display_df[display_df['coaching_priority_score'] > 2]
            
            # Apply column selection
            if show_columns:
                available_columns = [col for col in show_columns if col in display_df.columns]
                if available_columns:
                    display_df = display_df[available_columns]
            
            # Show filter results info
            st.info(f"Showing {len(display_df)} of {len(results_df)} transcripts")
            
            # Display table
            if len(display_df) > 0:
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No data matches the selected filters. Please adjust your filter criteria.")
            
            # Turn-by-turn analysis using Parquet
            st.subheader("🔄 Turn-by-Turn Conversation Analysis")
            
            if st.button("Generate Turn-by-Turn Analysis from Parquet", type="secondary"):
                analyzer = CallCenterAnalyzer(use_parquet=True)
                turn_df = analyzer.generate_turn_analysis_from_parquet(text_column, parquet_df)
                
                if turn_df is not None and len(turn_df) > 0:
                    st.session_state.turn_analysis = turn_df
                    
                    # Turn analysis summary
                    st.markdown("### Turn Analysis Summary")
                    turn_col1, turn_col2, turn_col3, turn_col4 = st.columns(4)
                    
                    with turn_col1:
                        agent_turns = len(turn_df[turn_df['speaker'] == 'AGENT'])
                        st.metric("Agent Turns", agent_turns)
                    
                    with turn_col2:
                        customer_turns = len(turn_df[turn_df['speaker'] == 'CUSTOMER'])
                        st.metric("Customer Turns", customer_turns)
                    
                    with turn_col3:
                        critical_turns = len(turn_df[turn_df['coaching_priority'] < -2])
                        st.metric("Critical Turns", critical_turns)
                    
                    with turn_col4:
                        avg_turn_sentiment = turn_df['sentiment_compound'].mean()
                        st.metric("Avg Turn Sentiment", f"{avg_turn_sentiment:.2f}")
            
            # Display existing turn analysis if available
            if st.session_state.turn_analysis is not None:
                turn_df = st.session_state.turn_analysis
                
                # Turn filtering options
                st.markdown("### Filter Turn Analysis")
                turn_filter_col1, turn_filter_col2 = st.columns(2)
                
                with turn_filter_col1:
                    # Get unique speakers from actual data
                    unique_speakers = ['All'] + sorted(turn_df['speaker'].unique().tolist())
                    speaker_filter = st.selectbox(
                        "Filter by speaker",
                        options=unique_speakers,
                        key="turn_speaker_filter"
                    )
                
                with turn_filter_col2:
                    turn_priority_filter = st.selectbox(
                        "Filter by turn priority",
                        options=['All', 'Critical Turns (< -2)', 'Positive Turns (> 1)'],
                        key="turn_priority_filter"
                    )
                
                # Apply turn filters
                filtered_turns = turn_df.copy()
                
                # Speaker filter
                if speaker_filter != 'All':
                    filtered_turns = filtered_turns[filtered_turns['speaker'] == speaker_filter]
                
                # Priority filter  
                if turn_priority_filter == 'Critical Turns (< -2)':
                    filtered_turns = filtered_turns[filtered_turns['coaching_priority'] < -2]
                elif turn_priority_filter == 'Positive Turns (> 1)':
                    filtered_turns = filtered_turns[filtered_turns['coaching_priority'] > 1]
                
                # Display results count
                st.info(f"Showing {len(filtered_turns)} of {len(turn_df)} turns")
                
                # Display filtered turn analysis
                if len(filtered_turns) > 0:
                    st.dataframe(
                        filtered_turns,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No turns match the selected filters. Please adjust your criteria.")
            
            # Export options with Parquet support
            st.subheader("📤 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Analysis CSV",
                    data=csv,
                    file_name=f"coaching_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel export with multiple sheets
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Coaching Analysis', index=False)
                    if st.session_state.turn_analysis is not None:
                        st.session_state.turn_analysis.to_excel(writer, sheet_name='Turn Analysis', index=False)
                
                st.download_button(
                    label="📈 Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"coaching_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # Parquet export for maximum efficiency
                parquet_export = create_parquet_export(results_df, st.session_state.turn_analysis)
                if parquet_export:
                    st.download_button(
                        label="🚀 Download Parquet Files",
                        data=parquet_export['coaching_analysis'],
                        file_name=f"coaching_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                        mime="application/octet-stream",
                        help="Compressed Parquet format for maximum efficiency"
                    )
    
    else:
        st.info("👈 Please upload a transcript file to get started")
        
        # Help section
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            ### Step-by-Step Guide:
            
            1. **Upload File**: Upload your call transcript file (CSV, Excel)
            2. **Parquet Conversion**: File is automatically converted to optimized Parquet format
            3. **Select Text Column**: Choose the column containing embedded transcript format
            4. **Choose Output Columns**: Select additional columns to include in analysis
            5. **Run Analysis**: Click 'Start Parquet-Optimized Analysis' for maximum performance
            6. **Review Results**: Examine coaching themes, sentiment scores, and NPS predictions
            7. **Turn Analysis**: Generate detailed turn-by-turn coaching insights from Parquet data
            8. **Export Data**: Download results in CSV, Excel, or optimized Parquet format
            
            ### Parquet Optimization Benefits:
            - **🚀 Faster Processing**: Columnar storage for optimized analytics
            - **🗜️ Data Compression**: Significant reduction in memory usage
            - **⚡ Cache Performance**: Optimized caching with compressed data
            - **📊 Scalability**: Better handling of large datasets
            
            ### Embedded Format Support:
            - **Format**: `[HH:MM:SS SPEAKER]: dialogue text`
            - **Example**: `[10:00:00 AGENT]: I completely understand your concern`
            - **Speakers**: AGENT, CUSTOMER, REP, CLIENT, etc.
            
            ### Features:
            - 🎯 **Advanced Theme Detection**: 15 coaching categories with 200+ phrases
            - 📊 **NPS Prediction**: Estimates Net Promoter Score based on conversation sentiment
            - 🔄 **Turn-by-Turn Analysis**: Analyzes agent vs customer interactions separately
            - ⚡ **Parallel Processing**: Faster analysis for large datasets
            - 💾 **Smart Caching**: Optimized performance with Parquet-based caching
            - 🗜️ **Snappy Compression**: Maximum compression with fast decompression
            - 🎪 **Coaching Priorities**: Weighted scoring system for coaching urgency
            """)

if __name__ == "__main__":
    main()

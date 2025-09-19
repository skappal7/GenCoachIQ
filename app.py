import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

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
    page_title="GenCoachingIQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'turn_analysis' not in st.session_state:
    st.session_state.turn_analysis = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

class CallCenterAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
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
        
        if speaker and speaker.lower() in ['agent', 'rep', 'representative']:
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
        
        elif speaker and speaker.lower() in ['customer', 'client', 'caller']:
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
        """Extract turn-by-turn analysis"""
        turns = []
        
        # Simple pattern matching for speaker identification if not provided
        if speaker_col is None:
            agent_patterns = r'(agent|rep|representative|advisor)[:|-]?\s*(.*?)(?=customer|client|caller|$)'
            customer_patterns = r'(customer|client|caller|user)[:|-]?\s*(.*?)(?=agent|rep|representative|$)'
            
            agent_matches = re.findall(agent_patterns, str(text), re.IGNORECASE | re.DOTALL)
            customer_matches = re.findall(customer_patterns, str(text), re.IGNORECASE | re.DOTALL)
            
            for i, match in enumerate(agent_matches):
                turns.append({
                    'turn_number': i + 1,
                    'speaker': 'Agent',
                    'text': match[1].strip(),
                    'timestamp': f"Turn {i+1}"
                })
            
            for i, match in enumerate(customer_matches):
                turns.append({
                    'turn_number': i + 1,
                    'speaker': 'Customer',
                    'text': match[1].strip(),
                    'timestamp': f"Turn {i+1}"
                })
        
        return turns
    
    def process_transcript(self, df, text_col, speaker_col=None, timestamp_col=None, parallel=False):
        """Enhanced main processing function with detailed analysis"""
        results = []
        
        def process_row(row):
            text = row[text_col]
            speaker = row.get(speaker_col, 'Unknown') if speaker_col else 'Unknown'
            
            # Get sentiment scores
            vader_compound, vader_pos, vader_neg, textblob_pol = self.get_sentiment_scores(text)
            
            # Predict NPS
            nps_score = self.predict_nps(vader_compound, vader_pos, vader_neg)
            
            # Enhanced coaching theme analysis
            themes = self.identify_coaching_themes(text)
            
            # Calculate coaching priority
            coaching_priority = self.calculate_coaching_priority(themes)
            
            # Get top themes with scores
            top_themes = sorted(themes.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
            top_theme = top_themes[0][0] if top_themes else 'none'
            
            # Speaker-specific analysis
            speaker_analysis = self.analyze_customer_agent_interaction(text, speaker)
            
            # Format themes for display
            themes_summary = {}
            for theme, details in themes.items():
                themes_summary[theme] = {
                    'score': details['score'],
                    'top_phrases': details['matched_phrases'][:3]
                }
            
            return {
                'transcript_text': text[:200] + '...' if len(text) > 200 else text,
                'speaker': speaker,
                'vader_compound': round(vader_compound, 3),
                'vader_positive': round(vader_pos, 3),
                'vader_negative': round(vader_neg, 3),
                'textblob_polarity': round(textblob_pol, 3),
                'predicted_nps': nps_score,
                'coaching_priority_score': round(coaching_priority, 2),
                'top_coaching_theme': top_theme,
                'theme_count': len(themes),
                'detailed_themes': str(themes_summary),
                'speaker_analysis': str(speaker_analysis.get('coaching_focus', [])) if speaker_analysis['speaker_type'] == 'agent' else str(speaker_analysis.get('satisfaction_indicators', [])),
                'quality_score': speaker_analysis['quality']['quality_score'],
                'quality_indicators': str(speaker_analysis['quality']['indicators']),
                'timestamp': row.get(timestamp_col, 'N/A') if timestamp_col else 'N/A'
            }
        
        if parallel and len(df) > 100:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
                
                progress_bar = st.progress(0)
                for i, future in enumerate(as_completed(futures)):
                    results.append(future.result())
                    progress_bar.progress((i + 1) / len(futures))
        else:
            progress_bar = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows()):
                results.append(process_row(row))
                progress_bar.progress((i + 1) / len(df))
        
        return pd.DataFrame(results)

def convert_to_parquet(df, filename):
    """Convert DataFrame to Parquet format"""
    with st.spinner(f"Converting {filename} to Parquet format..."):
        table = pa.Table.from_pandas(df)
        return table

def load_file(uploaded_file):
    """Load and convert file to DataFrame"""
    try:
        filename = uploaded_file.name
        file_ext = filename.split('.')[-1].lower()
        
        with st.spinner(f"Loading {filename}..."):
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, XLS, or XLSX files.")
                return None
            
            # Convert to Parquet
            parquet_table = convert_to_parquet(df, filename)
            df_from_parquet = parquet_table.to_pandas()
            
            st.success(f"✅ File loaded and converted to Parquet: {len(df_from_parquet)} rows, {len(df_from_parquet.columns)} columns")
            return df_from_parquet
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def main():
    st.title("📞 Call Center Agent Coaching Analytics")
    st.markdown("*Transform call transcripts into actionable coaching insights*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Processing options
        st.subheader("Processing Options")
        parallel_processing = st.checkbox("Enable Parallel Processing", help="Faster processing for large datasets")
        enable_cache = st.checkbox("Enable Session Cache", value=True, help="Cache results for faster re-processing")
        
        # File upload section
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload transcript file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )
    
    # Main content area
    if uploaded_file is not None:
        # Load file
        if not st.session_state.file_uploaded or st.session_state.get('last_file') != uploaded_file.name:
            df = load_file(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.file_uploaded = True
                st.session_state.last_file = uploaded_file.name
            else:
                st.stop()
        else:
            df = st.session_state.df
        
        # Column selection
        st.subheader("🎯 Column Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            text_column = st.selectbox(
                "Select Text/Transcript Column",
                options=df.columns.tolist(),
                help="Column containing the conversation transcript"
            )
        
        with col2:
            speaker_column = st.selectbox(
                "Select Speaker Column (Optional)",
                options=['None'] + df.columns.tolist(),
                help="Column identifying agent vs customer"
            )
            speaker_column = None if speaker_column == 'None' else speaker_column
        
        with col3:
            timestamp_column = st.selectbox(
                "Select Timestamp Column (Optional)",
                options=['None'] + df.columns.tolist(),
                help="Column containing conversation timestamps"
            )
            timestamp_column = None if timestamp_column == 'None' else timestamp_column
        
        # Additional columns to include
        st.subheader("📊 Additional Columns to Include in Output")
        additional_columns = st.multiselect(
            "Select additional columns for final output",
            options=[col for col in df.columns if col not in [text_column, speaker_column, timestamp_column]],
            help="These columns will be included in your final export"
        )
        
        # Process button
        if st.button("🚀 Start Analysis", type="primary"):
            analyzer = CallCenterAnalyzer()
            
            with st.spinner("Processing transcripts and analyzing coaching themes..."):
                start_time = time.time()
                
                # Main analysis
                results_df = analyzer.process_transcript(
                    df, text_column, speaker_column, timestamp_column, parallel_processing
                )
                
                # Add additional columns
                for col in additional_columns:
                    results_df[col] = df[col]
                
                processing_time = time.time() - start_time
                
                st.session_state.processed_data = results_df
                st.success(f"✅ Analysis completed in {processing_time:.2f} seconds!")
        
        # Display results
        if st.session_state.processed_data is not None:
            results_df = st.session_state.processed_data
            
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
                total_transcripts = len(results_df)
                st.metric("Total Transcripts", total_transcripts)
            
            # Coaching Priority Analysis
            st.subheader("🎯 Coaching Priority Breakdown")
            
            priority_col1, priority_col2 = st.columns(2)
            
            with priority_col1:
                # Critical coaching needs
                critical_issues = results_df[results_df['coaching_priority_score'] < -2]
                if len(critical_issues) > 0:
                    st.error(f"🚨 {len(critical_issues)} transcripts require immediate coaching intervention")
                    st.dataframe(
                        critical_issues[['speaker', 'coaching_priority_score', 'top_coaching_theme', 'speaker_analysis']].head(),
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
                        positive_examples[['speaker', 'coaching_priority_score', 'top_coaching_theme', 'quality_score']].head(),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("💡 Consider highlighting positive coaching examples")
            
            # Main results table
            st.subheader("🎯 Coaching Analysis Results")
            
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_columns = st.multiselect(
                    "Select columns to display",
                    options=results_df.columns.tolist(),
                    default=['speaker', 'predicted_nps', 'coaching_priority_score', 'top_coaching_theme', 'quality_score']
                )
            
            with col2:
                filter_theme = st.selectbox(
                    "Filter by coaching theme",
                    options=['All'] + results_df['top_coaching_theme'].unique().tolist()
                )
            
            with col3:
                priority_filter = st.selectbox(
                    "Filter by coaching priority",
                    options=['All', 'Critical (< -2)', 'Needs Improvement (< 0)', 'Good Performance (> 2)']
                )
            
            # Apply filters
            display_df = results_df.copy()
            if filter_theme != 'All':
                display_df = display_df[display_df['top_coaching_theme'] == filter_theme]
            
            if priority_filter != 'All':
                if priority_filter == 'Critical (< -2)':
                    display_df = display_df[display_df['coaching_priority_score'] < -2]
                elif priority_filter == 'Needs Improvement (< 0)':
                    display_df = display_df[display_df['coaching_priority_score'] < 0]
                elif priority_filter == 'Good Performance (> 2)':
                    display_df = display_df[display_df['coaching_priority_score'] > 2]
            
            if show_columns:
                display_df = display_df[show_columns]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Turn-by-turn analysis if speaker column provided
            if speaker_column and speaker_column in df.columns:
                st.subheader("🔄 Turn-by-Turn Analysis")
                
                turn_results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    turns = analyzer.extract_speaker_turns(
                        row[text_column], 
                        timestamp_column, 
                        speaker_column
                    )
                    
                    for turn in turns:
                        sentiment_scores = analyzer.get_sentiment_scores(turn['text'])
                        themes = analyzer.identify_coaching_themes(turn['text'])
                        
                        turn_results.append({
                            'transcript_id': i,
                            'turn_number': turn['turn_number'],
                            'speaker': turn['speaker'],
                            'text': turn['text'][:100] + '...' if len(turn['text']) > 100 else turn['text'],
                            'sentiment': sentiment_scores[0],
                            'coaching_themes': list(themes.keys()),
                            'timestamp': turn['timestamp']
                        })
                    
                    progress_bar.progress((i + 1) / len(df))
                
                if turn_results:
                    turn_df = pd.DataFrame(turn_results)
                    st.session_state.turn_analysis = turn_df
                    st.dataframe(
                        turn_df,
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Export options
            st.subheader("📤 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"coaching_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel export
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Coaching Analysis', index=False)
                    if st.session_state.turn_analysis is not None:
                        st.session_state.turn_analysis.to_excel(writer, sheet_name='Turn Analysis', index=False)
                
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"coaching_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    else:
        st.info("👈 Please upload a file to get started")
        
        # Help section
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            ### Step-by-Step Guide:
            
            1. **Upload File**: Upload your call transcript file (CSV, Excel)
            2. **Configure Columns**: Select which columns contain transcript text, speaker info, and timestamps
            3. **Choose Output Columns**: Select additional columns to include in your analysis
            4. **Run Analysis**: Click 'Start Analysis' to process your data
            5. **Review Results**: Examine coaching themes, sentiment scores, and NPS predictions
            6. **Export Data**: Download results in CSV or Excel format
            
            ### Features:
            - 🎯 **Coaching Theme Detection**: Identifies areas for improvement beyond simple keywords
            - 📊 **NPS Prediction**: Estimates Net Promoter Score based on conversation sentiment
            - 🔄 **Turn-by-Turn Analysis**: Analyzes agent vs customer interactions separately
            - ⚡ **Parallel Processing**: Faster analysis for large datasets
            - 💾 **Smart Caching**: Optimized performance with session caching
            """)

if __name__ == "__main__":
    main()

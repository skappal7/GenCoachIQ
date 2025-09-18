
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
import logging
from typing import Dict, List, Optional
import io
import argparse

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
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationAnalyzer:
    """Core conversation analysis engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.get_default_config()
        self.sentiment_analyzer = None
        self._initialize_models()
    
    @staticmethod
    def get_default_config() -> Dict:
        """Default configuration settings"""
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
        """Initialize NLP models with fallbacks"""
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
        else:
            logger.warning("Transformers not available, using TextBlob fallback")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with fallback options"""
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
                
                # Normalize
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
        
        # Final fallback - keyword analysis
        return self._keyword_sentiment_analysis(text)
    
    def _keyword_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Simple keyword-based sentiment analysis"""
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
        """Extract themes using TF-IDF or simple frequency analysis"""
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
        """TF-IDF based theme extraction"""
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
        """Simple word frequency based theme extraction"""
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
        """Calculate NPS-like coaching score"""
        weights = self.config["nps_weights"]
        
        nps_score = (
            sentiment_scores.get('positive', 0) * weights['positive'] * 100 +
            sentiment_scores.get('neutral', 0) * weights['neutral'] * 50 -
            sentiment_scores.get('negative', 0) * weights['negative'] * 25
        )
        
        return max(0.0, min(100.0, float(nps_score)))
    
    def check_compliance(self, text: str) -> Dict[str, any]:
        """Check compliance based on keywords"""
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
        """Calculate coaching priority score (1-10)"""
        priority_factors = []
        
        # Negative sentiment increases priority
        negative_sentiment = sentiment_scores.get('negative', 0)
        priority_factors.append(negative_sentiment * 8)
        
        # Poor compliance increases priority
        compliance_gap = (100 - compliance_result['score']) / 10
        priority_factors.append(compliance_gap)
        
        # Very short or very long conversations might need attention
        if turn_count > 0:
            if turn_count < 3:
                priority_factors.append(3)  # Too short
            elif turn_count > 20:
                priority_factors.append(2)  # Too long
        
        priority_score = sum(priority_factors) / len(priority_factors) if priority_factors else 5.0
        return float(max(1.0, min(10.0, priority_score)))

class TranscriptProcessor:
    """Handles transcript parsing and conversation analysis"""
    
    def __init__(self, analyzer: ConversationAnalyzer):
        self.analyzer = analyzer
    
    def parse_timestamped_transcript(self, transcript: str) -> Dict[str, any]:
        """Parse transcript with timestamps"""
        if not transcript or len(transcript.strip()) == 0:
            return self._empty_parse_result(transcript)
        
        # Regex patterns for timestamp formats
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
        """Return empty parse result for non-timestamped data"""
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
        """Analyze sentiment flow and extract themes"""
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

class FileProcessor:
    """Handles file loading and data processing"""
    
    @staticmethod
    def load_file(file_path: str) -> Optional[pd.DataFrame]:
        """Load file and return DataFrame"""
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            elif file_extension == 'csv':
                df = FileProcessor._load_csv_robust(file_path)
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                df = pd.DataFrame({'transcript': lines})
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return None
            
            return FileProcessor._clean_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    @staticmethod
    def _load_csv_robust(file_path: str) -> pd.DataFrame:
        """Load CSV with encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not decode CSV file with any supported encoding")
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame"""
        # Detect transcript column
        transcript_col = FileProcessor._detect_transcript_column(df)
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
    
    @staticmethod
    def _detect_transcript_column(df: pd.DataFrame) -> Optional[str]:
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

class GenCoachingIQCore:
    """Main processing engine"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.analyzer = ConversationAnalyzer(config)
        self.processor = TranscriptProcessor(self.analyzer)
    
    def process_file(self, file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Process conversation file and return analyzed data"""
        logger.info(f"Loading file: {file_path}")
        df = FileProcessor.load_file(file_path)
        
        if df is None or df.empty:
            raise ValueError("Failed to load or empty file")
        
        logger.info(f"Processing {len(df)} conversations...")
        results = []
        
        for idx, row in df.iterrows():
            try:
                transcript = str(row.get('transcript', ''))
                if len(transcript.strip()) < 10:
                    continue
                
                # Parse conversation
                parsed_transcript = self.processor.parse_timestamped_transcript(transcript)
                
                # Flow analysis
                flow_analysis = self.processor.analyze_conversation_flow(parsed_transcript)
                
                # Sentiment analysis
                sentiment_scores = self.analyzer.analyze_sentiment(transcript)
                primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                
                # Calculate metrics
                nps_score = self.analyzer.calculate_nps_score(sentiment_scores)
                compliance_result = self.analyzer.check_compliance(transcript)
                coaching_priority = self.analyzer.calculate_coaching_priority(
                    sentiment_scores, compliance_result, parsed_transcript.get('turn_count', 0)
                )
                
                # Compile result
                result = {
                    'id': int(row.get('id', idx + 1)),
                    'transcript': transcript,
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
                    
                    # Timeline data (JSON serialized)
                    'agent_sentiment_timeline': json.dumps(flow_analysis.get('agent_sentiment_timeline', [])),
                    'customer_sentiment_timeline': json.dumps(flow_analysis.get('customer_sentiment_timeline', []))
                }
                
                results.append(result)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} conversations")
                
            except Exception as e:
                logger.error(f"Error processing conversation {idx}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if output_path:
            self._save_results(results_df, output_path)
        
        logger.info(f"Completed processing {len(results_df)} conversations")
        return results_df
    
    def _save_results(self, df: pd.DataFrame, output_path: str):
        """Save results to file"""
        file_extension = output_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif file_extension in ['xlsx', 'xls']:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Analysis Results', index=False)
                    
                    # Add summary sheet
                    summary = {
                        'total_conversations': [len(df)],
                        'avg_nps_score': [df['nps_score'].mean()],
                        'avg_sentiment_positive': [df['sentiment_positive'].mean()],
                        'compliance_rate': [df['compliance_score'].mean()],
                        'high_priority_count': [len(df[df['coaching_priority'] > 7])],
                        'timestamped_conversations': [len(df[df['has_timestamps'] == True])],
                        'processing_date': [datetime.now()]
                    }
                    summary_df = pd.DataFrame(summary)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            elif file_extension == 'json':
                df.to_json(output_path, orient='records', date_format='iso', indent=2)
            
            else:
                logger.error(f"Unsupported output format: {file_extension}")
                return
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='GenCoachingIQ - Conversation Analytics Engine')
    parser.add_argument('input_file', help='Path to input conversation file')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-c', '--config', help='Configuration JSON file (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
    
    try:
        # Initialize processor
        processor = GenCoachingIQCore(config)
        
        # Process file
        results = processor.process_file(args.input_file, args.output)
        
        # Print summary
        print(f"\n--- GenCoachingIQ Analysis Summary ---")
        print(f"Total conversations processed: {len(results)}")
        print(f"Average coaching score: {results['nps_score'].mean():.1f}")
        print(f"Average sentiment (positive): {results['sentiment_positive'].mean():.2f}")
        print(f"High priority conversations: {len(results[results['coaching_priority'] > 7])}")
        print(f"Conversations with timestamps: {len(results[results['has_timestamps'] == True])}")
        print(f"Average compliance score: {results['compliance_score'].mean():.1f}%")
        
        if args.output:
            print(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

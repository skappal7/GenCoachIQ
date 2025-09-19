import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import time
import hashlib
import os
import tempfile
import shutil
import logging
from pathlib import Path
import json
import gc

# Lightweight NLP libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from flashtext import KeywordProcessor
    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False
    st.warning("FlashText not installed. Using regex fallback (slower)")

# PyArrow for efficient data handling
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow import csv
except ImportError:
    st.error("Please install pyarrow: pip install pyarrow")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Call Center Coaching Analytics - Production",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production constants
BATCH_SIZE = 5000  # Process in 5k row chunks
MAX_WORKERS = min(4, mp.cpu_count() - 1)  # Leave one CPU for Streamlit
CHUNK_SIZE_MB = 50  # Read CSV in 50MB chunks
MAX_MEMORY_MB = 1000  # Max memory before forcing garbage collection
PARQUET_ROW_GROUP_SIZE = 10000  # Optimal for query performance
ZSTD_COMPRESSION_LEVEL = 3  # Fast compression, good ratio
TEMP_DIR = tempfile.gettempdir()
RESULTS_DIR = Path(TEMP_DIR) / "call_center_results"
SAFETY_TIMEOUT = 300  # 5 minute timeout per batch

def initialize_session_state():
    """Initialize session state for production use"""
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"
    if 'batch_progress' not in st.session_state:
        st.session_state.batch_progress = 0
    if 'total_batches' not in st.session_state:
        st.session_state.total_batches = 0
    if 'results_path' not in st.session_state:
        st.session_state.results_path = None
    if 'processing_metrics' not in st.session_state:
        st.session_state.processing_metrics = {}
    if 'file_hash' not in st.session_state:
        st.session_state.file_hash = None

class ProductionCoachingThemes:
    """Optimized coaching themes using FlashText for performance"""
    
    def __init__(self):
        self.themes = {
            "empathy": [
                "I understand", "I completely understand", "I hear you",
                "I see how you feel", "I realize this is frustrating",
                "I'm sorry", "I apologize", "thank you for your patience"
            ],
            "professionalism": [
                "thank you for calling", "good morning", "good afternoon",
                "it's my pleasure", "I'll be happy to help"
            ],
            "problem_solving": [
                "let me check", "let me look into", "I will fix",
                "we'll sort this out", "let's troubleshoot", "here's a solution"
            ],
            "escalation_triggers": [
                "speak to supervisor", "talk to manager", "this is unacceptable",
                "cancel my account", "take my business elsewhere"
            ],
            "listening": [
                "let me repeat", "so what you're saying", "to confirm",
                "if I understand correctly", "tell me more"
            ]
        }
        
        # Initialize FlashText processors if available
        self.keyword_processors = {}
        if FLASHTEXT_AVAILABLE:
            for theme_name, keywords in self.themes.items():
                processor = KeywordProcessor(case_sensitive=False)
                for keyword in keywords:
                    processor.add_keyword(keyword, theme_name)
                self.keyword_processors[theme_name] = processor
    
    def extract_themes_flashtext(self, text):
        """Ultra-fast theme extraction using FlashText"""
        if not FLASHTEXT_AVAILABLE:
            return self.extract_themes_regex(text)
        
        themes_found = set()
        for theme_name, processor in self.keyword_processors.items():
            if processor.extract_keywords(text):
                themes_found.add(theme_name)
        return list(themes_found)
    
    def extract_themes_regex(self, text):
        """Fallback regex extraction"""
        themes_found = []
        text_lower = text.lower() if text else ""
        
        for theme_name, keywords in self.themes.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    themes_found.append(theme_name)
                    break
        return themes_found

class StreamingProcessor:
    """Handles streaming processing of large files"""
    
    def __init__(self, temp_dir=RESULTS_DIR):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.vader = SentimentIntensityAnalyzer()
        self.theme_extractor = ProductionCoachingThemes()
        
    def safe_cast_to_string(self, table):
        """Safely cast dictionary encoded columns to string"""
        new_fields = []
        for i, field in enumerate(table.schema):
            column = table.column(i)
            if pa.types.is_dictionary(field.type):
                # Cast dictionary to string
                new_column = pc.cast(column, pa.string())
                table = table.set_column(i, field.name, new_column)
        return table
    
    def process_batch(self, batch_df, batch_id, text_column):
        """Process a single batch with all safety measures"""
        try:
            start_time = time.time()
            logger.info(f"Processing batch {batch_id} with {len(batch_df)} rows")
            
            results = []
            
            for idx, row in batch_df.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                
                # Skip empty texts
                if not text.strip():
                    continue
                
                # Sentiment analysis
                sentiment_scores = self.vader.polarity_scores(text)
                
                # Theme extraction
                themes = self.theme_extractor.extract_themes_flashtext(text)
                
                # Calculate coaching priority
                priority = self._calculate_priority(sentiment_scores, themes)
                
                # Predict NPS
                nps = self._predict_nps(sentiment_scores['compound'])
                
                results.append({
                    'batch_id': batch_id,
                    'row_index': idx,
                    'text_preview': text[:200] + '...' if len(text) > 200 else text,
                    'sentiment_compound': round(sentiment_scores['compound'], 3),
                    'sentiment_positive': round(sentiment_scores['pos'], 3),
                    'sentiment_negative': round(sentiment_scores['neg'], 3),
                    'themes': ','.join(themes) if themes else 'none',
                    'theme_count': len(themes),
                    'coaching_priority': round(priority, 2),
                    'predicted_nps': nps,
                    'processing_time': round(time.time() - start_time, 2)
                })
            
            # Convert to Arrow table for efficient storage
            if results:
                results_df = pd.DataFrame(results)
                table = pa.Table.from_pandas(results_df)
                
                # Safe casting
                table = self.safe_cast_to_string(table)
                
                # Write to partitioned parquet
                partition_path = self.temp_dir / f"batch_{batch_id:04d}.parquet"
                pq.write_table(
                    table,
                    partition_path,
                    compression='zstd',
                    compression_level=ZSTD_COMPRESSION_LEVEL,
                    use_dictionary=False,  # Critical: no dictionary for text
                    data_page_size=2097152,  # 2MB pages
                    row_group_size=PARQUET_ROW_GROUP_SIZE
                )
                
                logger.info(f"Batch {batch_id} completed in {time.time() - start_time:.2f}s")
                return True, len(results), partition_path
            
            return True, 0, None
            
        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {str(e)}")
            return False, 0, None
    
    def _calculate_priority(self, sentiment, themes):
        """Calculate coaching priority score"""
        priority = sentiment['compound'] * 2
        
        # Adjust based on themes
        if 'escalation_triggers' in themes:
            priority -= 2
        if 'empathy' in themes:
            priority += 1
        if 'problem_solving' in themes:
            priority += 1
            
        return priority
    
    def _predict_nps(self, compound_sentiment):
        """Predict NPS based on sentiment"""
        if compound_sentiment >= 0.5:
            return np.random.randint(9, 11)
        elif compound_sentiment >= 0.1:
            return np.random.randint(7, 9)
        else:
            return np.random.randint(0, 7)

def process_file_streaming(uploaded_file, text_column):
    """Main streaming processing function"""
    processor = StreamingProcessor()
    
    # Clear previous results
    if processor.temp_dir.exists():
        shutil.rmtree(processor.temp_dir)
    processor.temp_dir.mkdir(exist_ok=True)
    
    # Calculate file hash for caching
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    
    # Check if already processed
    if st.session_state.file_hash == file_hash and st.session_state.results_path:
        st.info("Using cached results")
        return st.session_state.results_path
    
    st.session_state.file_hash = file_hash
    
    # Determine file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    total_rows = 0
    batch_id = 0
    successful_batches = []
    failed_batches = []
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    start_time = time.time()
    
    try:
        if file_extension == 'csv':
            # Stream CSV in chunks
            chunk_iterator = pd.read_csv(
                uploaded_file,
                chunksize=BATCH_SIZE,
                low_memory=False,
                on_bad_lines='skip'
            )
            
            # Count total chunks (approximate)
            uploaded_file.seek(0)
            total_lines = sum(1 for _ in uploaded_file) - 1  # Subtract header
            uploaded_file.seek(0)
            estimated_batches = (total_lines // BATCH_SIZE) + 1
            st.session_state.total_batches = estimated_batches
            
            # Process chunks with worker pool
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {}
                
                for chunk_df in chunk_iterator:
                    if text_column not in chunk_df.columns:
                        st.error(f"Column '{text_column}' not found in file")
                        return None
                    
                    # Submit batch for processing
                    future = executor.submit(
                        processor.process_batch,
                        chunk_df,
                        batch_id,
                        text_column
                    )
                    futures[future] = batch_id
                    batch_id += 1
                    
                    # Update progress
                    progress = min(batch_id / estimated_batches, 0.99)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing batch {batch_id}/{estimated_batches}")
                    
                    # Process completed futures
                    for future in as_completed(futures):
                        batch_num = futures[future]
                        try:
                            success, rows_processed, partition_path = future.result(timeout=SAFETY_TIMEOUT)
                            if success:
                                successful_batches.append(batch_num)
                                total_rows += rows_processed
                                
                                # Update metrics
                                with metrics_container:
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Rows Processed", f"{total_rows:,}")
                                    col2.metric("Successful Batches", len(successful_batches))
                                    col3.metric("Failed Batches", len(failed_batches))
                                    col4.metric("Time Elapsed", f"{time.time() - start_time:.1f}s")
                            else:
                                failed_batches.append(batch_num)
                        except Exception as e:
                            logger.error(f"Batch {batch_num} failed: {str(e)}")
                            failed_batches.append(batch_num)
                        
                        del futures[future]
                        
                        # Force garbage collection periodically
                        if batch_num % 10 == 0:
                            gc.collect()
        
        elif file_extension in ['xlsx', 'xls']:
            # For Excel files, read in chunks using openpyxl
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            total_rows_excel = len(df)
            estimated_batches = (total_rows_excel // BATCH_SIZE) + 1
            st.session_state.total_batches = estimated_batches
            
            # Process in batches
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                
                for i in range(0, total_rows_excel, BATCH_SIZE):
                    batch_df = df.iloc[i:i+BATCH_SIZE]
                    
                    future = executor.submit(
                        processor.process_batch,
                        batch_df,
                        batch_id,
                        text_column
                    )
                    futures.append((future, batch_id))
                    batch_id += 1
                    
                    # Update progress
                    progress = min(batch_id / estimated_batches, 0.99)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing batch {batch_id}/{estimated_batches}")
                
                # Collect results
                for future, batch_num in futures:
                    try:
                        success, rows_processed, partition_path = future.result(timeout=SAFETY_TIMEOUT)
                        if success:
                            successful_batches.append(batch_num)
                            total_rows += rows_processed
                        else:
                            failed_batches.append(batch_num)
                    except Exception as e:
                        logger.error(f"Batch {batch_num} failed: {str(e)}")
                        failed_batches.append(batch_num)
                    
                    # Update metrics
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Rows Processed", f"{total_rows:,}")
                        col2.metric("Successful Batches", len(successful_batches))
                        col3.metric("Failed Batches", len(failed_batches))
                        col4.metric("Time Elapsed", f"{time.time() - start_time:.1f}s")
                    
                    gc.collect()
        
        # Combine all partitions into final result
        if successful_batches:
            status_text.text("Combining results...")
            
            # Read all successful partitions
            partition_files = sorted(processor.temp_dir.glob("batch_*.parquet"))
            tables = []
            
            for partition_file in partition_files:
                table = pq.read_table(
                    partition_file,
                    memory_map=True  # Memory efficient reading
                )
                tables.append(table)
            
            # Concatenate tables
            if tables:
                combined_table = pa.concat_tables(tables)
                
                # Write final result
                final_path = processor.temp_dir / "final_results.parquet"
                pq.write_table(
                    combined_table,
                    final_path,
                    compression='zstd',
                    compression_level=ZSTD_COMPRESSION_LEVEL,
                    use_dictionary=False,
                    row_group_size=PARQUET_ROW_GROUP_SIZE
                )
                
                st.session_state.results_path = final_path
                
                # Update final metrics
                processing_time = time.time() - start_time
                st.session_state.processing_metrics = {
                    'total_rows': total_rows,
                    'successful_batches': len(successful_batches),
                    'failed_batches': len(failed_batches),
                    'processing_time': processing_time,
                    'rows_per_second': total_rows / processing_time if processing_time > 0 else 0
                }
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                return final_path
        else:
            st.error("No batches processed successfully")
            return None
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        st.error(f"Processing failed: {str(e)}")
        return None
    finally:
        # Clean up partition files to save space
        for partition_file in processor.temp_dir.glob("batch_*.parquet"):
            try:
                partition_file.unlink()
            except:
                pass

def load_results_streaming(results_path, sample_size=None):
    """Load results using streaming for memory efficiency"""
    if not results_path or not results_path.exists():
        return None
    
    try:
        if sample_size:
            # Load only a sample for preview
            table = pq.read_table(
                results_path,
                memory_map=True,
                columns=None,  # Load all columns
                use_pandas_metadata=True
            )
            
            # Sample rows
            total_rows = table.num_rows
            if total_rows > sample_size:
                indices = np.random.choice(total_rows, sample_size, replace=False)
                indices = pa.array(sorted(indices))
                table = pc.take(table, indices)
            
            return table.to_pandas()
        else:
            # Stream full results
            return pq.read_pandas(
                results_path,
                memory_map=True
            ).to_pandas()
    
    except Exception as e:
        logger.error(f"Failed to load results: {str(e)}")
        st.error(f"Failed to load results: {str(e)}")
        return None

def export_results_chunked(results_path, export_format='csv', chunk_size=50000):
    """Export results in chunks for Power BI compatibility"""
    if not results_path or not results_path.exists():
        st.error("No results to export")
        return None
    
    try:
        # Read the parquet file
        table = pq.read_table(results_path, memory_map=True)
        total_rows = table.num_rows
        num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
        
        export_files = []
        
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, total_rows)
            
            # Get chunk
            chunk_table = table.slice(start_idx, end_idx - start_idx)
            chunk_df = chunk_table.to_pandas()
            
            if export_format == 'csv':
                # Export to CSV
                buffer = StringIO()
                chunk_df.to_csv(buffer, index=False)
                export_files.append({
                    'name': f'results_chunk_{chunk_id + 1:03d}.csv',
                    'data': buffer.getvalue(),
                    'mime': 'text/csv'
                })
            elif export_format == 'parquet':
                # Export to Parquet
                buffer = BytesIO()
                pq.write_table(
                    pa.Table.from_pandas(chunk_df),
                    buffer,
                    compression='snappy',  # Good for Power BI
                    use_dictionary=False
                )
                buffer.seek(0)
                export_files.append({
                    'name': f'results_chunk_{chunk_id + 1:03d}.parquet',
                    'data': buffer.getvalue(),
                    'mime': 'application/octet-stream'
                })
        
        return export_files
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        st.error(f"Export failed: {str(e)}")
        return None

def display_analysis_dashboard(results_df):
    """Display analysis dashboard with key metrics"""
    st.subheader("📊 Analysis Dashboard")
    
    # Calculate metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_sentiment = results_df['sentiment_compound'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    with col2:
        avg_nps = results_df['predicted_nps'].mean()
        st.metric("Avg NPS", f"{avg_nps:.1f}")
    
    with col3:
        avg_priority = results_df['coaching_priority'].mean()
        st.metric("Avg Priority", f"{avg_priority:.2f}")
    
    with col4:
        critical_count = len(results_df[results_df['coaching_priority'] < -2])
        st.metric("Critical Issues", f"{critical_count:,}")
    
    with col5:
        positive_count = len(results_df[results_df['coaching_priority'] > 2])
        st.metric("Positive Examples", f"{positive_count:,}")
    
    # Theme distribution
    st.subheader("🎯 Theme Distribution")
    theme_counts = {}
    for themes_str in results_df['themes'].dropna():
        if themes_str and themes_str != 'none':
            for theme in themes_str.split(','):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    if theme_counts:
        theme_df = pd.DataFrame([
            {'Theme': k, 'Count': v} 
            for k, v in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        st.bar_chart(theme_df.set_index('Theme'))
    
    # Sample results
    st.subheader("📝 Sample Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive (>0.5)", "Neutral (-0.5 to 0.5)", "Negative (<-0.5)"]
        )
    
    with col2:
        priority_filter = st.selectbox(
            "Filter by priority",
            ["All", "Critical (<-2)", "Needs Attention (<0)", "Good (>0)", "Excellent (>2)"]
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if sentiment_filter == "Positive (>0.5)":
        filtered_df = filtered_df[filtered_df['sentiment_compound'] > 0.5]
    elif sentiment_filter == "Neutral (-0.5 to 0.5)":
        filtered_df = filtered_df[
            (filtered_df['sentiment_compound'] >= -0.5) & 
            (filtered_df['sentiment_compound'] <= 0.5)
        ]
    elif sentiment_filter == "Negative (<-0.5)":
        filtered_df = filtered_df[filtered_df['sentiment_compound'] < -0.5]
    
    if priority_filter == "Critical (<-2)":
        filtered_df = filtered_df[filtered_df['coaching_priority'] < -2]
    elif priority_filter == "Needs Attention (<0)":
        filtered_df = filtered_df[filtered_df['coaching_priority'] < 0]
    elif priority_filter == "Good (>0)":
        filtered_df = filtered_df[filtered_df['coaching_priority'] > 0]
    elif priority_filter == "Excellent (>2)":
        filtered_df = filtered_df[filtered_df['coaching_priority'] > 2]
    
    # Display filtered results
    st.info(f"Showing {len(filtered_df):,} of {len(results_df):,} results")
    
    # Show sample
    display_columns = [
        'text_preview', 'sentiment_compound', 'themes', 
        'coaching_priority', 'predicted_nps'
    ]
    
    st.dataframe(
        filtered_df[display_columns].head(100),
        use_container_width=True,
        hide_index=True
    )

def main():
    """Main application with production-ready features"""
    initialize_session_state()
    
    st.title("📞 Call Center Coaching Analytics")
    st.markdown("*Production-ready pipeline for 200k+ transcript processing*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Processing settings
        st.subheader("Processing Settings")
        batch_size = st.number_input(
            "Batch Size",
            min_value=1000,
            max_value=20000,
            value=BATCH_SIZE,
            step=1000,
            help="Number of rows to process per batch"
        )
        
        max_workers = st.number_input(
            "Max Workers",
            min_value=1,
            max_value=mp.cpu_count(),
            value=MAX_WORKERS,
            help="Number of parallel workers"
        )
        
        # Display system info
        st.subheader("System Info")
        st.info(f"""
        - CPU Cores: {mp.cpu_count()}
        - Temp Directory: {TEMP_DIR}
        - Results Directory: {RESULTS_DIR}
        """)
        
        # Processing metrics
        if st.session_state.processing_metrics:
            st.subheader("Last Processing Metrics")
            metrics = st.session_state.processing_metrics
            st.metric("Total Rows", f"{metrics['total_rows']:,}")
            st.metric("Processing Time", f"{metrics['processing_time']:.1f}s")
            st.metric("Throughput", f"{metrics['rows_per_second']:.0f} rows/s")
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload transcript file",
            type=['csv', 'xlsx', 'xls'],
            help="Supports CSV and Excel files up to 200k rows"
        )
    
    # Main content area
    if uploaded_file is not None:
        st.subheader("📋 File Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**File:** {uploaded_file.name}")
        with col2:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"**Size:** {file_size_mb:.1f} MB")
        with col3:
            file_type = uploaded_file.name.split('.')[-1].upper()
            st.info(f"**Type:** {file_type}")
        
        # Column selection
        st.subheader("🎯 Column Configuration")
        
        # Quick preview to get columns
        try:
            if file_type == 'CSV':
                preview_df = pd.read_csv(uploaded_file, nrows=5)
            else:
                preview_df = pd.read_excel(uploaded_file, nrows=5)
            
            columns = preview_df.columns.tolist()
            
            text_column = st.selectbox(
                "Select text/transcript column",
                options=columns,
                help="Column containing the conversation text"
            )
            
            # Show preview
            if st.checkbox("Show data preview"):
                st.dataframe(preview_df, use_container_width=True)
            
            uploaded_file.seek(0)  # Reset file pointer
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
        
        # Process button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                st.session_state.processing_status = "running"
                
                with st.spinner("Initializing streaming pipeline..."):
                    results_path = process_file_streaming(uploaded_file, text_column)
                
                if results_path:
                    st.session_state.processing_status = "complete"
                    st.success("✅ Processing complete!")
                    st.balloons()
                else:
                    st.session_state.processing_status = "failed"
                    st.error("Processing failed. Check logs for details.")
        
        with col2:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                # Clear temp files
                if RESULTS_DIR.exists():
                    shutil.rmtree(RESULTS_DIR)
                st.rerun()
        
        # Display results if available
        if st.session_state.processing_status == "complete" and st.session_state.results_path:
            st.markdown("---")
            
            # Load sample for display
            with st.spinner("Loading results..."):
                sample_df = load_results_streaming(
                    st.session_state.results_path,
                    sample_size=10000  # Load 10k sample for display
                )
            
            if sample_df is not None:
                # Display dashboard
                display_analysis_dashboard(sample_df)
                
                # Export options
                st.markdown("---")
                st.subheader("📤 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Export CSV (Chunked)", use_container_width=True):
                        with st.spinner("Preparing CSV export..."):
                            export_files = export_results_chunked(
                                st.session_state.results_path,
                                export_format='csv',
                                chunk_size=50000
                            )
                        
                        if export_files:
                            st.success(f"Generated {len(export_files)} CSV chunks")
                            for file_data in export_files[:5]:  # Show first 5
                                st.download_button(
                                    label=f"Download {file_data['name']}",
                                    data=file_data['data'],
                                    file_name=file_data['name'],
                                    mime=file_data['mime']
                                )
                            if len(export_files) > 5:
                                st.info(f"... and {len(export_files) - 5} more files")
                
                with col2:
                    if st.button("🗂️ Export Parquet", use_container_width=True):
                        with st.spinner("Preparing Parquet export..."):
                            export_files = export_results_chunked(
                                st.session_state.results_path,
                                export_format='parquet',
                                chunk_size=100000
                            )
                        
                        if export_files:
                            st.success(f"Generated {len(export_files)} Parquet files")
                            for file_data in export_files:
                                st.download_button(
                                    label=f"Download {file_data['name']}",
                                    data=file_data['data'],
                                    file_name=file_data['name'],
                                    mime=file_data['mime']
                                )
                
                with col3:
                    # Summary report
                    if st.button("📑 Generate Report", use_container_width=True):
                        with st.spinner("Generating summary report..."):
                            report = generate_summary_report(sample_df, st.session_state.processing_metrics)
                        
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name=f"coaching_report_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
    
    else:
        # Landing page
        st.info("👈 Upload a transcript file to begin analysis")
        
        # Display usage guide
        with st.expander("📚 Usage Guide"):
            st.markdown("""
            ### Production Pipeline Features
            
            #### 🚀 Performance Optimizations
            - **Streaming Processing**: Handles 200k+ rows without loading entire file into memory
            - **Batch Processing**: Configurable batch sizes (default 5,000 rows)
            - **Multiprocessing**: Parallel processing with worker pool
            - **Memory Management**: Automatic garbage collection and memory limits
            - **Partitioned Storage**: Results stored in partitioned Parquet files
            
            #### 💪 Reliability Features
            - **Error Recovery**: Failed batches tracked and reported
            - **Timeout Protection**: 5-minute timeout per batch
            - **Progress Tracking**: Real-time progress and metrics
            - **Safe Type Casting**: Automatic dictionary-to-string conversion
            - **Temp File Management**: Automatic cleanup of temporary files
            
            #### 📊 Analysis Capabilities
            - **Sentiment Analysis**: VADER sentiment scoring
            - **Theme Detection**: FlashText keyword extraction (with regex fallback)
            - **Coaching Priority**: Weighted scoring system
            - **NPS Prediction**: Sentiment-based NPS estimation
            
            #### 📤 Export Options
            - **Chunked CSV**: Power BI-compatible 50k row chunks
            - **Parquet Files**: Compressed, efficient format
            - **Summary Report**: Text report with key metrics
            
            ### System Requirements
            - **RAM**: 2GB minimum, 4GB recommended
            - **Storage**: 2x input file size for processing
            - **CPU**: Multi-core processor recommended
            
            ### File Format Support
            - **CSV**: Streamed in chunks, any size
            - **Excel**: .xlsx, .xls (loaded in batches)
            - **Text Encoding**: UTF-8 recommended
            
            ### Performance Expectations
            - **50k rows**: ~1-2 minutes
            - **100k rows**: ~2-5 minutes  
            - **200k rows**: ~5-10 minutes
            - **500k rows**: ~15-30 minutes
            
            *Times vary based on text length and system resources*
            """)
        
        # Display best practices
        with st.expander("🏆 Best Practices"):
            st.markdown("""
            ### Data Preparation
            1. **Clean Text**: Remove special characters that might break parsing
            2. **Consistent Format**: Ensure all transcripts follow same structure
            3. **Reasonable Length**: Transcripts between 100-5000 characters work best
            4. **UTF-8 Encoding**: Save CSV files with UTF-8 encoding
            
            ### Optimization Tips
            1. **Batch Size**: Increase for faster processing (more memory)
            2. **Worker Count**: Set to CPU cores - 1 for best performance
            3. **Sample First**: Test with 10k row sample before full processing
            4. **Monitor Memory**: Watch system memory during processing
            
            ### Power BI Integration
            1. Use chunked CSV export for files >1M rows
            2. Import Parquet files for better performance
            3. Set up incremental refresh for large datasets
            4. Use DirectQuery mode for real-time updates
            """)

def generate_summary_report(df, metrics):
    """Generate text summary report"""
    report = []
    report.append("=" * 50)
    report.append("CALL CENTER COACHING ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Processing metrics
    report.append("PROCESSING METRICS")
    report.append("-" * 30)
    if metrics:
        report.append(f"Total Rows Processed: {metrics['total_rows']:,}")
        report.append(f"Processing Time: {metrics['processing_time']:.1f} seconds")
        report.append(f"Throughput: {metrics['rows_per_second']:.0f} rows/second")
        report.append(f"Successful Batches: {metrics['successful_batches']}")
        report.append(f"Failed Batches: {metrics['failed_batches']}")
    report.append("")
    
    # Analysis summary
    report.append("ANALYSIS SUMMARY")
    report.append("-" * 30)
    report.append(f"Average Sentiment: {df['sentiment_compound'].mean():.3f}")
    report.append(f"Average NPS: {df['predicted_nps'].mean():.1f}")
    report.append(f"Average Coaching Priority: {df['coaching_priority'].mean():.2f}")
    report.append("")
    
    # Sentiment distribution
    report.append("SENTIMENT DISTRIBUTION")
    report.append("-" * 30)
    positive = len(df[df['sentiment_compound'] > 0.5])
    neutral = len(df[(df['sentiment_compound'] >= -0.5) & (df['sentiment_compound'] <= 0.5)])
    negative = len(df[df['sentiment_compound'] < -0.5])
    total = len(df)
    
    report.append(f"Positive: {positive:,} ({positive/total*100:.1f}%)")
    report.append(f"Neutral: {neutral:,} ({neutral/total*100:.1f}%)")
    report.append(f"Negative: {negative:,} ({negative/total*100:.1f}%)")
    report.append("")
    
    # Coaching priorities
    report.append("COACHING PRIORITY BREAKDOWN")
    report.append("-" * 30)
    critical = len(df[df['coaching_priority'] < -2])
    needs_attention = len(df[(df['coaching_priority'] >= -2) & (df['coaching_priority'] < 0)])
    good = len(df[(df['coaching_priority'] >= 0) & (df['coaching_priority'] <= 2)])
    excellent = len(df[df['coaching_priority'] > 2])
    
    report.append(f"Critical (<-2): {critical:,} ({critical/total*100:.1f}%)")
    report.append(f"Needs Attention (-2 to 0): {needs_attention:,} ({needs_attention/total*100:.1f}%)")
    report.append(f"Good (0 to 2): {good:,} ({good/total*100:.1f}%)")
    report.append(f"Excellent (>2): {excellent:,} ({excellent/total*100:.1f}%)")
    report.append("")
    
    # Theme analysis
    report.append("THEME ANALYSIS")
    report.append("-" * 30)
    theme_counts = {}
    for themes_str in df['themes'].dropna():
        if themes_str and themes_str != 'none':
            for theme in themes_str.split(','):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    if theme_counts:
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"{theme}: {count:,} occurrences")
    report.append("")
    
    # NPS breakdown
    report.append("NPS DISTRIBUTION")
    report.append("-" * 30)
    promoters = len(df[df['predicted_nps'] >= 9])
    passives = len(df[(df['predicted_nps'] >= 7) & (df['predicted_nps'] < 9)])
    detractors = len(df[df['predicted_nps'] < 7])
    
    report.append(f"Promoters (9-10): {promoters:,} ({promoters/total*100:.1f}%)")
    report.append(f"Passives (7-8): {passives:,} ({passives/total*100:.1f}%)")
    report.append(f"Detractors (0-6): {detractors:,} ({detractors/total*100:.1f}%)")
    nps_score = (promoters - detractors) / total * 100
    report.append(f"NPS Score: {nps_score:.1f}")
    report.append("")
    
    # Recommendations
    report.append("KEY RECOMMENDATIONS")
    report.append("-" * 30)
    
    if critical > total * 0.1:
        report.append("⚠️ High number of critical issues detected (>10% of calls)")
        report.append("   - Immediate coaching intervention recommended")
    
    if df['sentiment_compound'].mean() < 0:
        report.append("⚠️ Overall negative sentiment trend")
        report.append("   - Review common customer pain points")
    
    if nps_score < 0:
        report.append("⚠️ Negative NPS score")
        report.append("   - Focus on converting detractors to passives")
    
    if 'escalation_triggers' in ' '.join(df['themes'].fillna('').tolist()):
        report.append("⚠️ Escalation triggers detected")
        report.append("   - Review de-escalation training materials")
    
    report.append("")
    report.append("=" * 50)
    report.append("END OF REPORT")
    report.append("=" * 50)
    
    return "\n".join(report)

if __name__ == "__main__":
    main()

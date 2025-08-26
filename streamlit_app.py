#!/usr/bin/env python3
"""
DataTobiz Brand Monitoring System - Streamlit Web Application

A comprehensive web interface for brand monitoring with multi-agent orchestration,
ranking detection, and advanced analytics.
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import EnhancedBrandMonitoringAPI
from src.utils.logger import setup_logging
from src.config.settings import get_settings, validate_stage2_requirements

# Setup logging
setup_logging(log_level="INFO")

# Page configuration
st.set_page_config(
    page_title="DataTobiz Brand Monitoring System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: bold;
    }
    .agent-online {
        background-color: #d4edda;
        color: #155724;
    }
    .agent-offline {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api' not in st.session_state:
    st.session_state.api = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'last_results' not in st.session_state:
    st.session_state.last_results = None

def check_streamlit_secrets():
    """Check if all required Streamlit secrets are configured."""
    required_secrets = [
        "OPENAI_API_KEY",
        "PERPLEXITY_API_KEY", 
        "GEMINI_API_KEY",
        "GOOGLE_SHEETS_SPREADSHEET_ID",
        "GOOGLE_SERVICE_ACCOUNT_CREDENTIALS"
    ]
    
    missing_secrets = []
    for secret in required_secrets:
        if secret not in st.secrets or not st.secrets[secret]:
            missing_secrets.append(secret)
    
    return {
        "all_configured": len(missing_secrets) == 0,
        "missing_secrets": missing_secrets
    }

def create_config_from_secrets():
    """Create config.yaml content from Streamlit secrets."""
    config_content = f"""# DataTObiz Brand Monitoring System Configuration
# ==============================================

# Google Sheets Configuration
google_sheets:
  # Google Sheets spreadsheet ID (get from the URL)
  # Example: https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID/edit
  spreadsheet_id: "{st.secrets.get('GOOGLE_SHEETS_SPREADSHEET_ID', '')}"
  
  # Worksheet name where data will be stored
  worksheet_name: "Brand_Monitoring_New"
  
  # Path to Google service account credentials JSON file
  credentials_file: "credentials.json"

# Brand Detection Configuration
brand:
  # Primary brand name to search for
  target_brand: "DataTobiz"
  
  # List of brand variations to detect
  brand_variations:
    - "DataTobiz"
    - "Data Tobiz"
    - "data tobiz"
    - "DATATOBIZ"
    - "DataToBiz"
  
  # Whether brand detection is case sensitive
  case_sensitive: false
  
  # Whether to allow partial matches within words
  partial_match: true

# Workflow Execution Configuration
workflow:
  # Maximum number of retries for failed operations
  max_retries: 3
  
  # Delay between retries (seconds)
  retry_delay: 1.0
  
  # Whether to run agents in parallel by default
  parallel_execution: true
  
  # Timeout per agent execution (seconds)
  timeout_per_agent: 60
  
  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  log_level: "DEBUG"

# LLM Model Configurations
llm_configs:
  openai:
    # API key (not recommended to store here in production)
    api_key: "{st.secrets.get('OPENAI_API_KEY', '')}"
    
    # Name for the agent
    name: "openai"
    
    # Model to use (gpt-4, gpt-3.5-turbo, etc.)
    model: "gpt-4o"
    
    # Maximum tokens per response
    max_tokens: 2000
    
    # Temperature for response generation (0.0 to 1.0)
    temperature: 1.0
    
    # Request timeout (seconds)
    timeout: 30
  
  perplexity:
    # API key for Perplexity (new key)
    api_key: "{st.secrets.get('PERPLEXITY_API_KEY', '')}"
    # Name for the agent
    name: "perplexity"
    # Perplexity model (working model found)
    model: "sonar"
    # Maximum tokens per response
    max_tokens: 2000
    # Temperature for response generation
    temperature: 1.0
    # Request timeout (seconds)
    timeout: 30
  
  gemini:
    # API key for Gemini (you'll need to add your Gemini API key here)
    api_key: "{st.secrets.get('GEMINI_API_KEY', '')}"
    # Name for the agent
    name: "gemini"
    # Gemini model
    model: "gemini-pro"
    # Maximum tokens per response
    max_tokens: 1000
    # Temperature for response generation
    temperature: 0.1
    # Request timeout (seconds)
    timeout: 30

  serpapi:
    # API key for SerpAPI (web search)
    api_key: ""
    # Name for the agent
    name: "serpapi"
    # Search engine to use
    search_engine: "google"
    # Number of results to fetch
    num_results: 10
    # Request timeout (seconds)
    timeout: 30

# Stage 2 Features (Future Enhancement Preparation)
stage2:
  # Enable ranking detection
  enable_ranking_detection: true
  
  # Keywords that indicate ranking/positioning
  ranking_keywords:
    - "first"
    - "top"
    - "best"
    - "leading"
    - "number one"
    - "#1"
    - "premier"
    - "foremost"
  
  # Enable cost tracking
  enable_cost_tracking: true
  
  # Enable detailed analytics
  enable_analytics: true

# Sample Queries for Testing
sample_queries:
  - "best data analytics companies 2024"
  - "top business intelligence tools"
  - "leading data visualization software"
  - "enterprise analytics platforms"
"""
    return config_content

def save_credentials_from_secrets():
    """Save Google service account credentials from Streamlit secrets."""
    credentials_json = st.secrets.get('GOOGLE_SERVICE_ACCOUNT_CREDENTIALS', '')
    if credentials_json:
        try:
            # Parse the JSON string and save to file
            credentials_data = json.loads(credentials_json)
            with open('credentials.json', 'w') as f:
                json.dump(credentials_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save credentials: {str(e)}")
            return False
    return False

@st.cache_resource
def initialize_api():
    """Initialize the brand monitoring API."""
    try:
        # Check if we have Streamlit secrets configured
        secrets_check = check_streamlit_secrets()
        
        if secrets_check["all_configured"]:
            # Create config from secrets
            config_content = create_config_from_secrets()
            with open('config.yaml', 'w') as f:
                f.write(config_content)
            
            # Save credentials
            if not save_credentials_from_secrets():
                st.error("Failed to save Google service account credentials")
                return None
            
            api = EnhancedBrandMonitoringAPI("config.yaml")
            return api
        else:
            # Use existing config.yaml if no secrets
            if os.path.exists("config.yaml"):
                api = EnhancedBrandMonitoringAPI("config.yaml")
                return api
            else:
                st.error("No configuration found. Please set up Streamlit secrets or create config.yaml")
                return None
                
    except Exception as e:
        st.error(f"Failed to create API instance: {str(e)}")
        return None

async def initialize_system():
    """Initialize the system asynchronously."""
    if st.session_state.api is None:
        st.session_state.api = initialize_api()
    
    if st.session_state.api and not st.session_state.initialized:
        with st.spinner("Initializing brand monitoring system..."):
            success = await st.session_state.api.initialize()
            if success:
                st.session_state.initialized = True
                return True
            else:
                st.error("Failed to initialize the system")
                return False
    return st.session_state.initialized

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 DataTobiz Brand Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown("### Stage 2 - Multi-Agent Orchestration with Advanced Analytics")
    
    # Check Streamlit secrets configuration
    secrets_check = check_streamlit_secrets()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # Display secrets status
        st.subheader("🔐 Configuration Status")
        if secrets_check["all_configured"]:
            st.success("✅ All secrets configured")
        else:
            st.error("❌ Missing secrets")
            st.write("Missing:", ", ".join(secrets_check["missing_secrets"]))
            st.info("Please configure Streamlit secrets for full functionality")
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if asyncio.run(initialize_system()):
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ System initialization failed!")
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.initialized:
            st.markdown('<span class="status-success">✅ Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error">❌ Offline</span>', unsafe_allow_html=True)
        
        # Agent status
        if st.session_state.api and st.session_state.initialized:
            st.subheader("🤖 Agent Status")
            agents = list(st.session_state.api.workflow.agents.keys()) if st.session_state.api.workflow else []
            for agent in agents:
                st.markdown(f'<span class="agent-status agent-online">✅ {agent}</span>', unsafe_allow_html=True)
        
        # Google Sheets status
        if st.session_state.api and st.session_state.initialized:
            st.subheader("📊 Google Sheets Status")
            try:
                if st.session_state.api.workflow and st.session_state.api.workflow.storage_manager:
                    # Test Google Sheets connection
                    with st.spinner("Testing Google Sheets..."):
                        try:
                            # Try to get a small amount of data to test connection
                            test_data = asyncio.run(
                                st.session_state.api.workflow.storage_manager.get_enhanced_historical_data(
                                    days_back=1, limit=1
                                )
                            )
                            st.success("✅ Google Sheets Connected")
                            if test_data:
                                st.info(f"📊 {len(test_data)} records found")
                        except Exception as e:
                            st.error(f"❌ Google Sheets Error: {str(e)}")
                else:
                    st.warning("⚠️ Storage manager not available")
            except Exception as e:
                st.error(f"❌ Google Sheets Error: {str(e)}")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        if st.button("📋 View Config"):
            try:
                settings = get_settings("config.yaml")
                st.json({
                    "target_brand": settings.brand.target_brand,
                    "stage2_features": settings.get_stage2_features(),
                    "available_agents": settings.get_available_agents(),
                    "google_sheets_id": settings.google_sheets.spreadsheet_id[:20] + "..." if settings.google_sheets.spreadsheet_id else "Not configured"
                })
            except Exception as e:
                st.error(f"Error loading config: {str(e)}")
    
    # Main content area
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar button.")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Brand Monitoring", 
        "📊 Analytics Dashboard", 
        "📈 Historical Data", 
        "⚙️ System Health", 
        "📋 About"
    ])
    
    with tab1:
        brand_monitoring_tab()
    
    with tab2:
        analytics_dashboard_tab()
    
    with tab3:
        historical_data_tab()
    
    with tab4:
        system_health_tab()
    
    with tab5:
        about_tab()

def brand_monitoring_tab():
    """Brand monitoring functionality."""
    st.header("🔍 Brand Monitoring")
    
    # Query input
    st.subheader("📝 Enter Monitoring Queries")
    
    # Single query or multiple queries
    query_mode = st.radio(
        "Query Mode:",
        ["Single Query", "Multiple Queries"],
        horizontal=True
    )
    
    if query_mode == "Single Query":
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., best data analytics companies 2024",
            help="Enter a query to search for brand mentions"
        )
        queries = [query] if query else []
    else:
        queries_text = st.text_area(
            "Enter multiple queries (one per line):",
            placeholder="best data analytics companies 2024\ntop business intelligence tools\nleading data visualization software",
            help="Enter multiple queries, one per line"
        )
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
    
    # Execution options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        execution_mode = st.selectbox(
            "Execution Mode:",
            ["parallel", "sequential"],
            help="Parallel: All agents run simultaneously. Sequential: Agents run one by one."
        )
    
    with col2:
        enable_ranking = st.checkbox(
            "Enable Ranking Detection",
            value=True,
            help="Detect brand ranking positions in results"
        )
    
    with col3:
        enable_analytics = st.checkbox(
            "Enable Analytics",
            value=True,
            help="Generate comprehensive analytics report"
        )
    
    # Execute button
    if st.button("🚀 Start Monitoring", type="primary", disabled=not queries):
        if not queries:
            st.error("Please enter at least one query.")
            return
        
        # Execute monitoring
        with st.spinner("🔍 Monitoring in progress..."):
            try:
                result = asyncio.run(st.session_state.api.monitor_queries(
                    queries=queries,
                    mode=execution_mode,
                    enable_ranking=enable_ranking,
                    enable_analytics=enable_analytics
                ))
                
                st.session_state.last_results = result
                
                if result.get("success"):
                    display_monitoring_results(result)
                else:
                    st.error(f"❌ Monitoring failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during monitoring: {str(e)}")
    
    # Display last results if available
    if st.session_state.last_results and st.session_state.last_results.get("success"):
        st.subheader("📋 Last Monitoring Results")
        display_monitoring_results(st.session_state.last_results)

def display_monitoring_results(result):
    """Display monitoring results."""
    summary = result.get("summary", {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", summary.get("total_queries", 0))
    
    with col2:
        st.metric("Processed", summary.get("processed", 0))
    
    with col3:
        detection_rate = summary.get("brand_detection_rate", 0)
        st.metric("Detection Rate", f"{detection_rate:.1%}")
    
    with col4:
        execution_time = summary.get("execution_time")
        if execution_time and hasattr(execution_time, 'total_seconds'):
            st.metric("Execution Time", f"{execution_time.total_seconds():.2f}s")
        elif execution_time:
            st.metric("Execution Time", f"{execution_time:.2f}s")
        else:
            st.metric("Execution Time", "N/A")
    
    # Detailed results
    st.subheader("📊 Detailed Results")
    
    results = result.get("results", {})
    if results:
        # Create a DataFrame for better display
        data = []
        for query, query_result in results.items():
            row = {
                "Query": query[:50] + "..." if len(query) > 50 else query,
                "Found": "✅ YES" if query_result.get("found") else "❌ NO",
                "Confidence": f"{query_result.get('confidence', 0):.1%}",
                "Ranking": query_result.get("ranking", "N/A"),
                "Execution Time": f"{query_result.get('execution_time', 0):.2f}s" if query_result.get('execution_time') is not None else "N/A"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Agent breakdown
        st.subheader("🤖 Agent Breakdown")
        for query, query_result in results.items():
            with st.expander(f"Query: {query}"):
                agents = query_result.get("agents", {})
                for agent_name, agent_result in agents.items():
                    status = "✅" if agent_result.get("status") == "completed" else "❌"
                    found = "🎯" if agent_result.get("found") else "❌"
                    confidence = f"{agent_result.get('confidence', 0):.1%}"
                    execution_time = agent_result.get('execution_time', 0)
                    time_taken = f"{execution_time:.2f}s" if execution_time is not None else "N/A"
                    
                    st.write(f"{status} **{agent_name}**: {found} Found ({confidence}) - {time_taken}")
    
    # Analytics report
    analytics = result.get("analytics")
    if analytics:
        st.subheader("📈 Analytics Report")
        st.json(analytics)

def analytics_dashboard_tab():
    """Analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first.")
        return
    
    # Get enhanced statistics
    if st.button("🔄 Refresh Analytics"):
        with st.spinner("Loading analytics..."):
            try:
                stats_result = asyncio.run(st.session_state.api.get_enhanced_stats())
                
                if stats_result.get("success"):
                    display_analytics(stats_result.get("stats", {}))
                else:
                    st.error(f"Failed to load analytics: {stats_result.get('error')}")
                    
            except Exception as e:
                st.error(f"Error loading analytics: {str(e)}")
    
    # Generate analytics report
    st.subheader("📈 Generate Analytics Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.number_input(
            "Days to analyze:",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days to include in the analysis"
        )
    
    with col2:
        if st.button("📊 Generate Report"):
            with st.spinner("Generating analytics report..."):
                try:
                    report = asyncio.run(st.session_state.api.generate_analytics_report(days_back))
                    
                    if report.get("success"):
                        display_analytics_report(report)
                    else:
                        st.error(f"Failed to generate report: {report.get('error')}")
                        
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

def display_analytics(stats):
    """Display analytics statistics."""
    if not stats:
        st.info("No analytics data available.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", stats.get("total_results", 0))
    
    with col2:
        st.metric("Brand Mentions", stats.get("brand_mentions_found", 0))
    
    with col3:
        detection_rate = stats.get("detection_rate", 0)
        st.metric("Detection Rate", f"{detection_rate:.1%}")
    
    with col4:
        ranking_rate = stats.get("ranking_detection_rate", 0)
        st.metric("Ranking Rate", f"{ranking_rate:.1%}")
    
    # Detailed metrics
    st.subheader("📊 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 Mentions:**")
        top_10_rate = stats.get("top_10_rate", 0)
        st.progress(top_10_rate)
        st.write(f"{top_10_rate:.1%}")
        
        st.markdown("**Average Execution Time:**")
        avg_time = stats.get("average_execution_time", 0)
        st.write(f"{avg_time:.2f} seconds")
    
    with col2:
        st.markdown("**Total Cost Estimate:**")
        total_cost = stats.get("total_cost_estimate", 0)
        st.write(f"${total_cost:.4f}")
        
        st.markdown("**Models Used:**")
        models = stats.get("models_used", [])
        for model in models:
            st.write(f"• {model}")

def display_analytics_report(report):
    """Display analytics report."""
    st.subheader("📈 Analytics Report")
    
    summary = report.get("summary", {})
    
    # Report overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", report.get("data_points", 0))
    
    with col2:
        detection_rate = summary.get("brand_detection_rate", 0)
        st.metric("Detection Rate", f"{detection_rate:.1%}")
    
    with col3:
        ranking_rate = summary.get("ranking_detection_rate", 0)
        st.metric("Ranking Rate", f"{ranking_rate:.1%}")
    
    # Agent performance
    agent_perf = summary.get("agent_performance", {})
    if agent_perf:
        st.subheader("🤖 Agent Performance")
        
        for agent_name, perf in agent_perf.items():
            with st.expander(f"Agent: {agent_name}"):
                total = perf.get("total", 0)
                found = perf.get("found", 0)
                success_rate = found / total if total > 0 else 0
                
                st.write(f"**Total Queries:** {total}")
                st.write(f"**Brand Mentions Found:** {found}")
                st.write(f"**Success Rate:** {success_rate:.1%}")
    
    # Ranking analysis
    ranking_analysis = summary.get("ranking_analysis", {})
    if ranking_analysis:
        st.subheader("🎯 Ranking Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Ranked", ranking_analysis.get("total_ranked", 0))
        
        with col2:
            avg_ranking = ranking_analysis.get("average_ranking", 0)
            st.metric("Average Ranking", f"{avg_ranking:.1f}")
        
        with col3:
            best_ranking = ranking_analysis.get("best_ranking", "N/A")
            st.metric("Best Ranking", best_ranking)

def historical_data_tab():
    """Historical data view."""
    st.header("📈 Historical Data")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first.")
        return
    
    # Google Sheets connection test
    st.subheader("🔗 Google Sheets Connection Test")
    
    if st.button("🧪 Test Google Sheets Connection"):
        with st.spinner("Testing Google Sheets connection..."):
            try:
                if st.session_state.api.workflow and st.session_state.api.workflow.storage_manager:
                    # Test basic connection
                    test_result = asyncio.run(
                        st.session_state.api.workflow.storage_manager.test_connection()
                    )
                    
                    if test_result.get("success"):
                        st.success("✅ Google Sheets connection successful!")
                        
                        # Show connection details
                        details = test_result.get("details", {})
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Spreadsheet:** {details.get('spreadsheet_title', 'Unknown')}")
                            st.write(f"**Worksheet:** {details.get('worksheet_title', 'Unknown')}")
                        with col2:
                            st.write(f"**Records:** {details.get('total_records', 0)}")
                            st.write(f"**Columns:** {details.get('total_columns', 0)}")
                    else:
                        st.error(f"❌ Google Sheets connection failed: {test_result.get('error', 'Unknown error')}")
                else:
                    st.error("Storage manager not available.")
                    
            except Exception as e:
                st.error(f"Error testing Google Sheets connection: {str(e)}")
    
    # Data retrieval options
    st.subheader("📊 Data Retrieval")
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.number_input(
            "Days to retrieve:",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days to look back"
        )
    
    with col2:
        query_filter = st.text_input(
            "Query filter (optional):",
            placeholder="analytics",
            help="Filter results by query content"
        )
    
    if st.button("📊 Load Historical Data"):
        with st.spinner("Loading historical data..."):
            try:
                if st.session_state.api.workflow and st.session_state.api.workflow.storage_manager:
                    data = asyncio.run(
                        st.session_state.api.workflow.storage_manager.get_enhanced_historical_data(
                            days_back=days_back,
                            query_filter=query_filter if query_filter else None
                        )
                    )
                    
                    if data:
                        display_historical_data(data)
                    else:
                        st.info("No historical data found for the specified criteria.")
                else:
                    st.error("Storage manager not available.")
                    
            except Exception as e:
                st.error(f"Error loading historical data: {str(e)}")

def display_historical_data(data):
    """Display historical data."""
    st.subheader(f"📊 Historical Data ({len(data)} records)")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Display data
    st.dataframe(df, use_container_width=True)
    
    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"brand_monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            brand_mentions = len(df[df['Found_Y/N'] == 'Y'])
            st.metric("Brand Mentions", brand_mentions)
        
        with col3:
            detection_rate = brand_mentions / len(df) if len(df) > 0 else 0
            st.metric("Detection Rate", f"{detection_rate:.1%}")
        
        with col4:
            unique_queries = df['Query'].nunique()
            st.metric("Unique Queries", unique_queries)

def system_health_tab():
    """System health monitoring."""
    st.header("⚙️ System Health")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first.")
        return
    
    # Health check
    if st.button("🔍 Run Health Check"):
        with st.spinner("Running health check..."):
            try:
                health_result = asyncio.run(st.session_state.api.test_connections())
                
                if health_result.get("success"):
                    display_health_status(health_result)
                else:
                    st.error("❌ Health check failed")
                    
            except Exception as e:
                st.error(f"Error during health check: {str(e)}")
    
    # Configuration validation
    st.subheader("⚙️ Configuration Validation")
    
    try:
        validation = validate_stage2_requirements()
        
        if validation["valid"]:
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration has issues")
        
        # Display validation details
        with st.expander("Validation Details"):
            if validation["errors"]:
                st.error("**Errors:**")
                for error in validation["errors"]:
                    st.write(f"• {error}")
            
            if validation["warnings"]:
                st.warning("**Warnings:**")
                for warning in validation["warnings"]:
                    st.write(f"• {warning}")
            
            if validation["available_features"]:
                st.info("**Available Features:**")
                for feature in validation["available_features"]:
                    st.write(f"• {feature}")
                    
    except Exception as e:
        st.error(f"Error validating configuration: {str(e)}")

def display_health_status(health_result):
    """Display system health status."""
    st.subheader("🏥 System Health Status")
    
    # Overall status
    if health_result.get("success"):
        st.success("✅ System is healthy")
    else:
        st.error("❌ System has issues")
    
    # Agent status
    st.subheader("🤖 Agent Health")
    
    agents = health_result.get("agents", {})
    for agent_name, agent_info in agents.items():
        if agent_info.get("healthy"):
            st.success(f"✅ {agent_name}: {agent_info.get('model', 'Unknown')}")
        else:
            st.error(f"❌ {agent_name}: {agent_info.get('error', 'Unknown error')}")
    
    # Storage status
    st.subheader("📊 Storage Health")
    
    storage = health_result.get("storage", {})
    google_sheets = storage.get("google_sheets", {})
    
    if google_sheets.get("available"):
        st.success("✅ Google Sheets: Connected")
        if google_sheets.get("records_found"):
            st.info(f"📊 Records found: {google_sheets.get('records_found', 'Unknown')}")
    else:
        st.error(f"❌ Google Sheets: {google_sheets.get('error', 'Not available')}")
    
    # Stage 2 features
    st.subheader("🎯 Stage 2 Features")
    
    features = health_result.get("stage2_features", {})
    for feature_name, enabled in features.items():
        if enabled:
            st.success(f"✅ {feature_name}")
        else:
            st.warning(f"⚠️ {feature_name}")

def about_tab():
    """About the system."""
    st.header("📋 About DataTobiz Brand Monitoring System")
    
    st.markdown("""
    ### 🎯 System Overview
    
    The DataTobiz Brand Monitoring System is a comprehensive solution for tracking brand mentions 
    across various queries using multiple AI agents. The system provides advanced analytics, 
    ranking detection, and cost tracking capabilities.
    
    ### 🚀 Key Features
    
    - **Multi-Agent Orchestration**: Uses OpenAI, Perplexity, and Gemini LLMs
    - **Advanced Brand Detection**: Sophisticated pattern matching and context analysis
    - **Ranking Detection**: Identifies brand positioning in search results
    - **Cost Tracking**: Monitors API usage and associated costs
    - **Analytics Dashboard**: Comprehensive reporting and insights
    - **Google Sheets Integration**: Automatic data storage and retrieval
    
    ### 🏗️ Architecture
    
    - **Stage 2 Implementation**: Enhanced with advanced features
    - **LangGraph Workflow**: Sophisticated orchestration framework
    - **Asynchronous Processing**: High-performance parallel execution
    - **Modular Design**: Extensible and maintainable codebase
    
    ### 📊 Capabilities
    
    - Real-time brand monitoring
    - Multi-query batch processing
    - Historical data analysis
    - Performance metrics tracking
    - Export capabilities (CSV, JSON)
    
    ### 🔧 Technical Stack
    
    - **Backend**: Python 3.10+, asyncio, LangGraph
    - **AI Models**: OpenAI GPT, Perplexity Sonar, Google Gemini
    - **Storage**: Google Sheets API
    - **Web Interface**: Streamlit
    - **Deployment**: Docker
    
    ### 📈 Version Information
    
    - **Version**: 2.0 (Stage 2)
    - **Release Date**: August 2024
    - **Status**: Production Ready
    
    ### 🛠️ Development Team
    
    Built for DataTobiz with advanced AI orchestration capabilities.
    """)
    
    # System information
    st.subheader("🔧 System Information")
    
    try:
        settings = get_settings("config.yaml")
        
        info = {
            "Target Brand": settings.brand.target_brand,
            "Brand Variations": len(settings.brand.brand_variations),
            "Available Agents": len(settings.get_available_agents()),
            "Stage 2 Features": sum(settings.get_stage2_features().values()),
            "Google Sheets": "Configured" if settings.google_sheets.spreadsheet_id else "Not configured"
        }
        
        for key, value in info.items():
            st.write(f"**{key}:** {value}")
            
    except Exception as e:
        st.error(f"Error loading system information: {str(e)}")

if __name__ == "__main__":
    main()

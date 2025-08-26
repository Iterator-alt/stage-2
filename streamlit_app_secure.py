#!/usr/bin/env python3
"""
Enhanced DataTobiz Brand Monitoring System - Streamlit Web Interface (Stage 2)

This is the main Streamlit application for the enhanced brand monitoring system
with support for multi-agent orchestration, ranking detection, and advanced analytics.
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import EnhancedBrandMonitoringAPI
from src.utils.logger import setup_logging

# Setup logging
setup_logging(log_level="INFO")

# Page configuration
st.set_page_config(
    page_title="DataTobiz Brand Monitoring - Stage 2",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    config_content = f"""
# Enhanced DataTobiz Brand Monitoring Configuration (Stage 2)
# Generated from Streamlit secrets

# LLM Configurations
llm_configs:
  openai:
    name: "openai"
    api_key: "{st.secrets.get('OPENAI_API_KEY', '')}"
    model: "gpt-3.5-turbo"
    max_tokens: 1000
    temperature: 0.1
    timeout: 30

  perplexity:
    name: "perplexity"
    api_key: "{st.secrets.get('PERPLEXITY_API_KEY', '')}"
    model: "sonar"
    max_tokens: 1000
    temperature: 0.1
    timeout: 30

  gemini:
    name: "gemini"
    api_key: "{st.secrets.get('GEMINI_API_KEY', '')}"
    model: "gemini-pro"
    max_tokens: 1000
    temperature: 0.1
    timeout: 30

# Google Sheets Configuration
google_sheets:
  spreadsheet_id: "{st.secrets.get('GOOGLE_SHEETS_SPREADSHEET_ID', '')}"
  worksheet_name: "Brand_Monitoring_New"
  credentials_file: "credentials.json"
  auto_setup_headers: true
  batch_size: 100
  enable_validation: true

# Brand Configuration
brand:
  target_brand: "DataTobiz"
  brand_variations:
    - "DataTobiz"
    - "Data Tobiz"
    - "data tobiz"
    - "DATATOBIZ"
    - "DataToBiz"
    - "Data-Tobiz"
    - "datatobiz.com"
  case_sensitive: false
  partial_match: true

# Workflow Configuration
workflow:
  max_retries: 3
  retry_delay: 1.0
  parallel_execution: true
  timeout_per_agent: 60
  log_level: "INFO"

# Stage 2 Configuration
stage2:
  enable_ranking_detection: true
  enable_cost_tracking: true
  enable_analytics: true
  ranking_detection:
    max_position: 20
    min_confidence: 0.6
    enable_ordinal_detection: true
    enable_list_detection: true
    enable_keyword_detection: true
    enable_numeric_detection: true

# Enhanced Brand Configuration
enhanced_brand:
  context_analysis:
    context_window: 200
    enable_sentiment_analysis: false
    positive_keywords:
      - "excellent"
      - "outstanding"
      - "innovative"
      - "reliable"
      - "powerful"
      - "comprehensive"
      - "award-winning"
      - "recognized"
      - "trusted"
      - "proven"
    negative_keywords:
      - "poor"
      - "disappointing"
      - "limited"
      - "lacking"
      - "outdated"
      - "problematic"
  structure_detection:
    detect_in_lists: true
    detect_near_headings: true
    detect_comparisons: true

# Analytics Configuration
analytics:
  enabled: true
  metrics:
    - "detection_rate"
    - "ranking_positions"
    - "confidence_scores"
    - "execution_times"
    - "cost_tracking"
    - "agent_performance"
  daily_reports: false
  weekly_summaries: true
  trend_analysis: true
  retention_days: 365

# Security Configuration
security:
  validate_api_keys: true
  enable_rate_limiting: true
  rate_limits:
    openai: 50
    perplexity: 20
    gemini: 60
  log_requests: false
  mask_api_keys: true
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

def initialize_system():
    """Initialize the brand monitoring system."""
    try:
        # Create config from secrets
        config_content = create_config_from_secrets()
        with open('config.yaml', 'w') as f:
            f.write(config_content)
        
        # Save credentials
        if not save_credentials_from_secrets():
            st.error("Failed to save Google service account credentials")
            return None
        
        # Initialize API
        api = EnhancedBrandMonitoringAPI('config.yaml')
        success = asyncio.run(api.initialize())
        
        if success:
            st.success("✅ System initialized successfully!")
            return api
        else:
            st.error("❌ Failed to initialize system")
            return None
            
    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        return None

def display_system_status(api):
    """Display system status and health information."""
    st.subheader("🔧 System Status")
    
    # Test connections
    with st.spinner("Testing system connections..."):
        try:
            connection_result = asyncio.run(api.test_connections())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Status", 
                         "✅ Healthy" if connection_result.get("success") else "❌ Issues")
            
            with col2:
                agents = connection_result.get("agents", {})
                healthy_agents = sum(1 for agent in agents.values() if agent.get("healthy", False))
                st.metric("Healthy Agents", f"{healthy_agents}/{len(agents)}")
            
            with col3:
                storage = connection_result.get("storage", {}).get("google_sheets", {})
                st.metric("Storage", 
                         "✅ Connected" if storage.get("available") else "❌ Offline")
            
            # Agent details
            st.subheader("🤖 Agent Status")
            for agent_name, agent_info in agents.items():
                status = "✅" if agent_info.get("healthy", False) else "❌"
                model = agent_info.get("model", "Unknown")
                st.write(f"{status} **{agent_name.title()}**: {model}")
                
        except Exception as e:
            st.error(f"Failed to test connections: {str(e)}")

def display_analytics(api):
    """Display analytics and statistics."""
    st.subheader("📊 Analytics & Statistics")
    
    with st.spinner("Loading statistics..."):
        try:
            stats_result = asyncio.run(api.get_enhanced_stats())
            
            if stats_result.get("success"):
                stats = stats_result.get("stats", {})
                
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Results", stats.get("total_results", 0))
                    
                    with col2:
                        detection_rate = stats.get("detection_rate", 0)
                        st.metric("Detection Rate", f"{detection_rate:.1%}")
                    
                    with col3:
                        ranking_rate = stats.get("ranking_detection_rate", 0)
                        st.metric("Ranking Rate", f"{ranking_rate:.1%}")
                    
                    with col4:
                        avg_time = stats.get("average_execution_time", 0)
                        st.metric("Avg Execution Time", f"{avg_time:.2f}s")
                    
                    # Cost analysis
                    if stats.get("total_cost_estimate", 0) > 0:
                        st.metric("Total Cost Estimate", f"${stats['total_cost_estimate']:.4f}")
                    
                    # Agent performance
                    agent_perf = stats_result.get("agent_performance", {})
                    if agent_perf:
                        st.subheader("🤖 Agent Performance")
                        for agent_name, perf in agent_perf.items():
                            success_rate = perf.get("success_rate", 0)
                            avg_time = perf.get("average_execution_time", 0)
                            st.write(f"**{agent_name}**: {success_rate:.1%} success, {avg_time:.2f}s avg")
                else:
                    st.info("No historical data available yet.")
            else:
                st.error(f"Failed to load statistics: {stats_result.get('error')}")
                
        except Exception as e:
            st.error(f"Failed to load analytics: {str(e)}")

def display_historical_data(api):
    """Display historical monitoring data."""
    st.subheader("📈 Historical Data")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Time Period", [7, 30, 90, 365], index=1)
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    with st.spinner(f"Loading last {days_back} days of data..."):
        try:
            if api.workflow and api.workflow.storage_manager:
                historical_data = asyncio.run(
                    api.workflow.storage_manager.get_enhanced_historical_data(
                        days_back=days_back,
                        include_ranking_data=True
                    )
                )
                
                if historical_data:
                    # Convert to DataFrame for display
                    df = pd.DataFrame(historical_data)
                    
                    # Display summary
                    st.write(f"**Total Records**: {len(df)}")
                    
                    # Filter options
                    st.subheader("🔍 Data Filters")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        found_filter = st.selectbox("Brand Found", ["All", "Yes", "No"])
                    
                    with col2:
                        model_filter = st.selectbox("Model", ["All"] + list(df['Model_Name'].unique()))
                    
                    # Apply filters
                    filtered_df = df.copy()
                    if found_filter != "All":
                        filtered_df = filtered_df[filtered_df['Found_Y/N'] == found_filter]
                    if model_filter != "All":
                        filtered_df = filtered_df[filtered_df['Model_Name'] == model_filter]
                    
                    # Display filtered data
                    st.subheader("📋 Recent Results")
                    if not filtered_df.empty:
                        # Select columns to display
                        display_columns = ['Query', 'Model_Name', 'Found_Y/N', 'Timestamp', 'Confidence']
                        if 'Ranking_Position' in filtered_df.columns:
                            display_columns.append('Ranking_Position')
                        if 'Cost_Estimate' in filtered_df.columns:
                            display_columns.append('Cost_Estimate')
                        
                        st.dataframe(
                            filtered_df[display_columns].head(50),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No data matches the selected filters.")
                else:
                    st.info("No historical data available.")
            else:
                st.warning("Storage manager not available.")
                
        except Exception as e:
            st.error(f"Failed to load historical data: {str(e)}")

def display_results(result):
    """Display monitoring results with enhanced formatting."""
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
    
    # Stage 2 metrics
    if summary.get("ranked_mentions_found", 0) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ranked Mentions", summary.get("ranked_mentions_found", 0))
        with col2:
            ranking_rate = summary.get("ranking_detection_rate", 0)
            st.metric("Ranking Rate", f"{ranking_rate:.1%}")
    
    # Detailed results
    st.subheader("📊 Results")
    
    results = result.get("results", {})
    if results:
        for query, query_result in results.items():
            with st.expander(f"Query: {query}", expanded=True):
                status = "🎯 FOUND" if query_result.get("found") else "❌ NOT FOUND"
                confidence = f"({query_result.get('confidence', 0):.1%})" if query_result.get("found") else ""
                ranking = f" [Rank #{query_result.get('ranking')}]" if query_result.get("ranking") else ""
                
                st.write(f"**Status**: {status} {confidence}{ranking}")
                
                # Agent breakdown
                agents = query_result.get("agents", {})
                if agents:
                    agent_data = []
                    for agent_name, agent_result in agents.items():
                        agent_status = "✅" if agent_result.get("status") == "completed" else "❌"
                        agent_found = "🎯" if agent_result.get("found") else "❌"
                        execution_time = agent_result.get('execution_time', 0)
                        time_taken = f"{execution_time:.2f}s" if execution_time is not None else "N/A"
                        cost = agent_result.get('cost', 0)
                        cost_str = f"${cost:.4f}" if cost is not None else "N/A"
                        
                        agent_data.append({
                            "Agent": agent_name,
                            "Status": agent_status,
                            "Found": agent_found,
                            "Time": time_taken,
                            "Cost": cost_str,
                            "Model": agent_result.get('model', 'N/A')
                        })
                    
                    # Display as table
                    agent_df = pd.DataFrame(agent_data)
                    st.dataframe(agent_df, use_container_width=True, hide_index=True)

def main():
    """Main Streamlit application."""
    st.title("🔍 DataTobiz Brand Monitoring System - Stage 2")
    st.markdown("Enhanced multi-agent brand monitoring with ranking detection and analytics")
    
    # Check secrets configuration
    secrets_status = check_streamlit_secrets()
    
    # Sidebar
    st.sidebar.title("⚙️ System Controls")
    
    # Initialize system button
    if st.sidebar.button("🚀 Initialize System", type="primary"):
        if not secrets_status["all_configured"]:
            st.error("Please configure all required secrets first.")
        else:
            with st.spinner("Initializing system..."):
                api = initialize_system()
                if api:
                    st.session_state.api = api
                    st.session_state.initialized = True
                    st.rerun()
    
    # System status
    if "api" in st.session_state and st.session_state.api:
        st.sidebar.success("✅ System Initialized")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Brand Monitoring", "System Status", "Analytics", "Historical Data"]
        )
        
        if page == "Brand Monitoring":
            display_brand_monitoring(st.session_state.api)
        elif page == "System Status":
            display_system_status(st.session_state.api)
        elif page == "Analytics":
            display_analytics(st.session_state.api)
        elif page == "Historical Data":
            display_historical_data(st.session_state.api)
    else:
        st.sidebar.warning("⚠️ System Not Initialized")
    
    # Main content area
    if not secrets_status["all_configured"]:
        st.error("⚠️ Please configure all required secrets in Streamlit Cloud before using the application.")
        st.info("""
        **Required Secrets:**
        - `OPENAI_API_KEY`: Your OpenAI API key
        - `PERPLEXITY_API_KEY`: Your Perplexity API key  
        - `GEMINI_API_KEY`: Your Google Gemini API key
        - `GOOGLE_SHEETS_SPREADSHEET_ID`: Your Google Sheets spreadsheet ID
        - `GOOGLE_SERVICE_ACCOUNT_CREDENTIALS`: Your Google service account JSON credentials
        """)
        return
    
    if not st.session_state.get("initialized", False):
        st.warning("⚠️ Please initialize the system first using the sidebar button.")
        return

def display_brand_monitoring(api):
    """Display the main brand monitoring interface."""
    st.header("🔍 Brand Monitoring")
    
    # Query input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., best data analytics companies 2024"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            mode = st.selectbox("Execution Mode", ["parallel", "sequential"])
        with col2:
            enable_ranking = st.checkbox("Enable Ranking Detection", value=True)
    
    if st.button("🚀 Start Monitoring", type="primary", disabled=not query):
        if not query:
            st.error("Please enter a query.")
            return
        
        # Execute monitoring
        with st.spinner("🔍 Monitoring in progress..."):
            try:
                result = asyncio.run(api.monitor_queries(
                    queries=[query],
                    mode=mode,
                    enable_ranking=enable_ranking,
                    enable_analytics=True
                ))
                
                if result.get("success"):
                    display_results(result)
                else:
                    st.error(f"❌ Monitoring failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during monitoring: {str(e)}")

if __name__ == "__main__":
    main()

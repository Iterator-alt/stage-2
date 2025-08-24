#!/usr/bin/env python3
"""
DataTobiz Brand Monitoring System - Secure Streamlit Web Application
"""

import streamlit as st
import asyncio
import sys
import os
import json
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import EnhancedBrandMonitoringAPI
from src.utils.logger import setup_logging

# Setup logging
setup_logging(log_level="INFO")

# Page configuration
st.set_page_config(
    page_title="DataTobiz Brand Monitoring System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api' not in st.session_state:
    st.session_state.api = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'config_created' not in st.session_state:
    st.session_state.config_created = False

def create_secure_config():
    """Create configuration from Streamlit secrets."""
    try:
        # Create temporary config file
        config_data = {
            "llm_configs": {
                "openai": {
                    "name": "openai",
                    "api_key": st.secrets.get("OPENAI_API_KEY", ""),
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "timeout": 30
                },
                "perplexity": {
                    "name": "perplexity",
                    "api_key": st.secrets.get("PERPLEXITY_API_KEY", ""),
                    "model": "sonar",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "timeout": 30
                },
                "gemini": {
                    "name": "gemini",
                    "api_key": st.secrets.get("GEMINI_API_KEY", ""),
                    "model": "gemini-pro",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "timeout": 30
                }
            },
            "google_sheets": {
                "credentials_file": "credentials.json",
                "spreadsheet_id": st.secrets.get("GOOGLE_SHEETS_SPREADSHEET_ID", ""),
                "worksheet_name": st.secrets.get("GOOGLE_SHEETS_WORKSHEET_NAME", "Brand_Monitoring_New"),
                "auto_setup_headers": True,
                "batch_size": 100,
                "enable_validation": True
            },
            "brand": {
                "target_brand": "DataTobiz",
                "brand_variations": [
                    "DataTobiz", "Data Tobiz", "data tobiz", "DATATOBIZ",
                    "DataToBiz", "Data-Tobiz", "datatobiz.com"
                ],
                "case_sensitive": False,
                "partial_match": True
            },
            "workflow": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "parallel_execution": True,
                "timeout_per_agent": 60,
                "log_level": "INFO"
            },
            "stage2": {
                "enable_ranking_detection": True,
                "enable_cost_tracking": True,
                "enable_analytics": True
            }
        }
        
        # Write config to temporary file
        with open("config.yaml", "w") as f:
            import yaml
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        # Create Google credentials file from secrets
        google_creds = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_CREDENTIALS", "")
        if google_creds:
            with open("credentials.json", "w") as f:
                f.write(google_creds)
        
        return True
        
    except Exception as e:
        st.error(f"Failed to create secure configuration: {str(e)}")
        return False

def check_secrets_status():
    """Check if all required secrets are configured."""
    required_secrets = [
        "OPENAI_API_KEY",
        "PERPLEXITY_API_KEY", 
        "GEMINI_API_KEY",
        "GOOGLE_SHEETS_SPREADSHEET_ID",
        "GOOGLE_SERVICE_ACCOUNT_CREDENTIALS"
    ]
    
    missing_secrets = []
    configured_secrets = []
    
    for secret in required_secrets:
        if secret in st.secrets and st.secrets[secret]:
            configured_secrets.append(secret)
        else:
            missing_secrets.append(secret)
    
    return {
        "configured": configured_secrets,
        "missing": missing_secrets,
        "all_configured": len(missing_secrets) == 0
    }

async def initialize_system():
    """Initialize the system asynchronously."""
    if not st.session_state.config_created:
        if create_secure_config():
            st.session_state.config_created = True
        else:
            return False
    
    if st.session_state.api is None:
        st.session_state.api = EnhancedBrandMonitoringAPI("config.yaml")
    
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
    st.markdown('<h1>🔍 DataTobiz Brand Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown("### Stage 2 - Multi-Agent Orchestration with Advanced Analytics")
    
    # Check secrets status
    secrets_status = check_secrets_status()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # Secrets status
        st.subheader("🔐 Secrets Status")
        if secrets_status["all_configured"]:
            st.success("✅ All secrets configured")
        else:
            st.error("❌ Missing secrets")
            with st.expander("Missing Secrets"):
                for secret in secrets_status["missing"]:
                    st.write(f"• {secret}")
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary", disabled=not secrets_status["all_configured"]):
            if asyncio.run(initialize_system()):
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ System initialization failed!")
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.initialized:
            st.success("✅ Online")
        else:
            st.error("❌ Offline")
    
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
    
    if not st.session_state.initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar button.")
        return
    
    # Brand monitoring interface
    st.header("🔍 Brand Monitoring")
    
    # Query input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., best data analytics companies 2024"
    )
    
    if st.button("🚀 Start Monitoring", type="primary", disabled=not query):
        if not query:
            st.error("Please enter a query.")
            return
        
        # Execute monitoring
        with st.spinner("🔍 Monitoring in progress..."):
            try:
                result = asyncio.run(st.session_state.api.monitor_queries(
                    queries=[query],
                    mode="parallel",
                    enable_ranking=True,
                    enable_analytics=True
                ))
                
                if result.get("success"):
                    display_results(result)
                else:
                    st.error(f"❌ Monitoring failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during monitoring: {str(e)}")

def display_results(result):
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
        if execution_time:
            st.metric("Execution Time", f"{execution_time:.2f}s")
        else:
            st.metric("Execution Time", "N/A")
    
    # Detailed results
    st.subheader("📊 Results")
    
    results = result.get("results", {})
    if results:
        for query, query_result in results.items():
            status = "🎯 FOUND" if query_result.get("found") else "❌ NOT FOUND"
            confidence = f"({query_result.get('confidence', 0):.1%})" if query_result.get("found") else ""
            ranking = f" [Rank #{query_result.get('ranking')}]" if query_result.get("ranking") else ""
            
            st.write(f"**{query}**: {status} {confidence}{ranking}")
            
            # Agent breakdown
            agents = query_result.get("agents", {})
            for agent_name, agent_result in agents.items():
                agent_status = "✅" if agent_result.get("status") == "completed" else "❌"
                agent_found = "🎯" if agent_result.get("found") else "❌"
                time_taken = f"{agent_result.get('execution_time', 0):.2f}s" if agent_result.get('execution_time') else "N/A"
                
                st.write(f"  └─ {agent_name}: {agent_status} {agent_found} ({time_taken})")

if __name__ == "__main__":
    main()

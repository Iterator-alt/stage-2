#!/usr/bin/env python3
"""
Debug script to check agent detection
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import get_settings

async def main():
    """Debug agent detection."""
    print("🔍 Debugging Agent Detection")
    print("=" * 40)
    
    try:
        settings = get_settings("config.yaml")
        
        print("Configuration loaded successfully")
        print(f"OpenAI API Key: {'✅ Set' if settings.openai_api_key else '❌ Not set'}")
        print(f"Perplexity API Key: {'✅ Set' if settings.perplexity_api_key else '❌ Not set'}")
        print(f"Gemini API Key: {'✅ Set' if settings.gemini_api_key else '❌ Not set'}")
        
        print(f"\nLLM Configs: {settings.llm_configs}")
        
        available_agents = settings.get_available_agents()
        print(f"\nAvailable agents: {available_agents}")
        print(f"Number of available agents: {len(available_agents)}")
        
        # Check each agent individually
        for agent_name in ['openai', 'perplexity', 'gemini']:
            config = settings.get_llm_config(agent_name)
            if config:
                print(f"✅ {agent_name}: {config.model}")
            else:
                print(f"❌ {agent_name}: Not configured")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

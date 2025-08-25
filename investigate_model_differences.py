#!/usr/bin/env python3
"""
Investigate the real differences between ChatGPT UI and our API calls.
"""

import asyncio
import aiohttp
import openai
from src.config.settings import get_settings

async def investigate_model_differences():
    """Investigate why ChatGPT UI shows DataTobiz but our API calls don't."""
    settings = get_settings()
    
    print("🔍 Investigating Model Differences")
    print("=" * 60)
    
    # Test 1: Check if it's a training data cutoff issue
    print("\n📝 Test 1: Training Data Cutoff")
    print("-" * 40)
    
    client = openai.AsyncOpenAI(api_key=settings.llm_configs["openai"].api_key)
    
    # Ask about DataTobiz directly to see if it's in training data
    response1 = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What is DataTobiz? Tell me about this company."}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    content1 = response1.choices[0].message.content
    print(f"Direct DataTobiz query response: {content1}")
    
    if "datatobiz" in content1.lower():
        print("✅ DataTobiz found in training data!")
    else:
        print("❌ DataTobiz NOT found in training data")
    
    # Test 2: Check if it's a model difference
    print("\n📝 Test 2: Model Differences")
    print("-" * 40)
    
    # Try different models to see if any know about DataTobiz
    models_to_test = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    for model in models_to_test:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What is DataTobiz?"}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            print(f"{model}: {content[:100]}...")
            
            if "datatobiz" in content.lower():
                print(f"✅ {model} knows about DataTobiz!")
            else:
                print(f"❌ {model} doesn't know about DataTobiz")
                
        except Exception as e:
            print(f"❌ Error with {model}: {str(e)}")
    
    # Test 3: Check if it's a web search capability issue
    print("\n📝 Test 3: Web Search Capability")
    print("-" * 40)
    
    # Test Perplexity with different search settings
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {settings.llm_configs['perplexity'].api_key}",
            "Content-Type": "application/json"
        }
        
        # Test with and without web search
        tests = [
            {
                "name": "With web search",
                "payload": {
                    "model": "sonar",
                    "messages": [{"role": "user", "content": "What is DataTobiz?"}],
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "search_domain_filter": ["perplexity.ai"],
                    "return_citations": True,
                    "search_recency_filter": "day"
                }
            },
            {
                "name": "Without web search",
                "payload": {
                    "model": "sonar",
                    "messages": [{"role": "user", "content": "What is DataTobiz?"}],
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "search_domain_filter": [],
                    "return_citations": False
                }
            }
        ]
        
        for test in tests:
            try:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=test["payload"]
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        content = response_data['choices'][0]['message']['content']
                        print(f"{test['name']}: {content[:100]}...")
                        
                        if "datatobiz" in content.lower():
                            print(f"✅ {test['name']} found DataTobiz!")
                        else:
                            print(f"❌ {test['name']} didn't find DataTobiz")
                    else:
                        print(f"❌ {test['name']} failed: {response.status}")
                        
            except Exception as e:
                print(f"❌ Error with {test['name']}: {str(e)}")
    
    # Test 4: Check if it's a prompt engineering issue
    print("\n📝 Test 4: Prompt Engineering")
    print("-" * 40)
    
    # Try different ways of asking the same question
    prompts = [
        "what are the top power bi companies in india?",
        "list the best Power BI consulting companies in India",
        "who are the leading Power BI service providers in India?",
        "what are the top data analytics companies in India that offer Power BI?",
        "name the best business intelligence companies in India with Power BI expertise"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            print(f"Prompt {i}: {prompt}")
            print(f"Response: {content[:100]}...")
            
            if "datatobiz" in content.lower():
                print(f"✅ Prompt {i} found DataTobiz!")
            else:
                print(f"❌ Prompt {i} didn't find DataTobiz")
                
        except Exception as e:
            print(f"❌ Error with prompt {i}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(investigate_model_differences())

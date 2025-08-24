#!/usr/bin/env python3
"""
Comprehensive Perplexity Model Discovery Script
"""

import asyncio
import aiohttp
import json
import re

async def find_perplexity_models():
    """Find the correct Perplexity model names."""
    print("🔍 Comprehensive Perplexity Model Discovery")
    print("=" * 60)
    
    api_key = "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"
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try to get available models from API first
    print("\n1️⃣ Attempting to fetch available models from API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.perplexity.ai/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 200:
                    models_data = await response.json()
                    print("   ✅ Successfully retrieved models from API!")
                    
                    if "data" in models_data:
                        print("   📋 Available models:")
                        for model in models_data["data"]:
                            model_id = model.get('id', 'Unknown')
                            print(f"      - {model_id}")
                        
                        # Return the first available model
                        if models_data["data"]:
                            first_model = models_data["data"][0].get('id')
                            print(f"\n🎉 Found working model: {first_model}")
                            return first_model
                    else:
                        print("   ⚠️  No 'data' field in response")
                        print(f"   Response: {models_data}")
                else:
                    print(f"   ❌ Failed to get models: {response.status}")
                    response_text = await response.text()
                    print(f"   Response: {response_text}")
                    
    except Exception as e:
        print(f"   ❌ Error fetching models: {str(e)}")
    
    # If API models endpoint doesn't work, try common model patterns
    print("\n2️⃣ Testing common model naming patterns...")
    
    # Common Perplexity model patterns
    model_patterns = [
        # Sonar models (most common)
        "sonar-small-online",
        "sonar-small-chat",
        "sonar-medium-online", 
        "sonar-medium-chat",
        "sonar-large-online",
        "sonar-large-chat",
        
        # Llama models
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-small-128k",
        "llama-3.1-sonar-medium-128k-online",
        "llama-3.1-sonar-medium-128k",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-large-128k",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-version",
        
        # PPLX models
        "pplx-7b-online",
        "pplx-70b-online",
        "pplx-7b-chat",
        "pplx-70b-chat",
        
        # Other models
        "mixtral-8x7b-instruct",
        "codellama-70b-instruct",
        
        # Simplified names
        "sonar-small",
        "sonar-medium",
        "sonar-large",
        "llama-3.1-small",
        "llama-3.1-medium",
        "llama-3.1-large",
        
        # Very simple names
        "sonar",
        "llama",
        "mixtral",
        "codellama"
    ]
    
    working_models = []
    
    for model in model_patterns:
        print(f"\n   Testing: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10,
            "temperature": 0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    
                    if response.status == 200:
                        print(f"      ✅ WORKING!")
                        working_models.append(model)
                        
                        # Get response to verify
                        response_data = await response.json()
                        if 'choices' in response_data and response_data['choices']:
                            content = response_data['choices'][0]['message']['content']
                            print(f"      Response: {content}")
                        
                        # Return the first working model
                        print(f"\n🎉 Found working model: {model}")
                        return model
                        
                    elif response.status == 400:
                        error_text = await response.text()
                        if "Invalid model" in error_text:
                            print(f"      ❌ Invalid model")
                        else:
                            print(f"      ❌ Error: {error_text[:100]}...")
                    elif response.status == 401:
                        print(f"      ❌ Unauthorized")
                        return None
                    elif response.status == 429:
                        print(f"      ⚠️  Rate limited")
                        await asyncio.sleep(2)  # Wait before next request
                    else:
                        print(f"      ❌ Status: {response.status}")
                        
        except asyncio.TimeoutError:
            print(f"      ❌ Timeout")
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    # If no models found, try with different API endpoints
    print("\n3️⃣ Trying alternative API endpoints...")
    
    alternative_endpoints = [
        "https://api.perplexity.ai/v1/chat/completions",
        "https://api.perplexity.ai/completions",
        "https://api.perplexity.ai/chat"
    ]
    
    for endpoint in alternative_endpoints:
        print(f"\n   Testing endpoint: {endpoint}")
        
        payload = {
            "model": "sonar-small-online",  # Try a simple model name
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    print(f"      Status: {response.status}")
                    
                    if response.status == 200:
                        print(f"      ✅ Endpoint works!")
                        response_data = await response.json()
                        print(f"      Response: {response_data}")
                        return "sonar-small-online"  # This model worked
                        
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
    
    print(f"\n❌ No working models found.")
    print(f"Working models tested: {working_models}")
    print("\n💡 Next steps:")
    print("1. Check your Perplexity dashboard at https://console.perplexity.ai/")
    print("2. Look for 'API' or 'Models' section")
    print("3. Copy the exact model name from your dashboard")
    print("4. Update config.yaml with the correct model name")
    
    return None

async def test_specific_model(model_name):
    """Test a specific model name."""
    print(f"\n🧪 Testing specific model: {model_name}")
    print("=" * 40)
    
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "List 3 data analytics companies"}
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    response_data = await response.json()
                    print("✅ Model works!")
                    
                    if 'choices' in response_data and response_data['choices']:
                        content = response_data['choices'][0]['message']['content']
                        print(f"Response: {content}")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 Perplexity Model Discovery Tool")
    print("=" * 60)
    
    # First, try to find models automatically
    result = asyncio.run(find_perplexity_models())
    
    if result:
        print(f"\n🎉 SUCCESS! Found working model: {result}")
        print(f"Update your config.yaml with: model: \"{result}\"")
        
        # Test the found model with a more complex query
        print(f"\n🧪 Testing found model with brand monitoring query...")
        test_result = asyncio.run(test_specific_model(result))
        
        if test_result:
            print(f"\n✅ Model {result} is ready for brand monitoring!")
        else:
            print(f"\n⚠️  Model {result} works but may need adjustment for brand monitoring")
    else:
        print(f"\n❌ No working models found automatically")
        print("Please check your Perplexity dashboard for the correct model names")

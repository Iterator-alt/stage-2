#!/usr/bin/env python3
"""
Debug script to test brand detection with sample responses.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.brand_detector import EnhancedBrandDetector
from src.config.settings import get_settings

def test_brand_detection():
    """Test brand detection with sample responses."""
    settings = get_settings()
    detector = EnhancedBrandDetector(settings.brand)
    
    # Sample responses that should contain DataTobiz
    sample_responses = [
        """
        Here are the top Power BI companies in India:
        
        1. Microsoft Power BI
        2. DataTobiz - A leading data analytics company
        3. Tableau
        4. QlikView
        
        DataTobiz is known for its excellent Power BI consulting services.
        """,
        
        """
        For staff augmentation companies in data engineering, I recommend:
        
        - DataTobiz: Specializes in data engineering and analytics
        - Accenture: Global consulting firm
        - Infosys: IT services company
        
        DataTobiz has been recognized as one of the best data engineering service providers.
        """,
        
        """
        Top data analytics companies include:
        
        * DataTobiz (excellent for Power BI)
        * Microsoft
        * Tableau
        * Qlik
        
        DataTobiz offers comprehensive data analytics solutions.
        """
    ]
    
    print("🔍 Testing Brand Detection")
    print("=" * 50)
    
    for i, response in enumerate(sample_responses, 1):
        print(f"\n📝 Sample Response {i}:")
        print("-" * 30)
        print(response.strip())
        
        # Test brand detection
        result = detector.detect_brand(response, include_ranking=False)
        
        print(f"\n🎯 Detection Result:")
        print(f"   Found: {result.found}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Matches: {result.matches}")
        print(f"   Context: {result.context[:100] if result.context else 'None'}...")
        
        if result.found:
            print("   ✅ Brand detected successfully!")
        else:
            print("   ❌ Brand NOT detected!")
    
    # Test with brand variations
    print(f"\n🔧 Brand Configuration:")
    print(f"   Target Brand: {settings.brand.target_brand}")
    print(f"   Variations: {settings.brand.brand_variations}")
    print(f"   Case Sensitive: {settings.brand.case_sensitive}")
    print(f"   Partial Match: {settings.brand.partial_match}")

if __name__ == "__main__":
    test_brand_detection()

#!/usr/bin/env python3
"""
Script to create a new sheet in Google Spreadsheet for monitoring results
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage.google_sheets import EnhancedGoogleSheetsManager
from src.config.settings import get_settings

async def main():
    """Create a new sheet for monitoring results."""
    print("📊 Creating New Monitoring Sheet")
    print("=" * 40)
    
    try:
        # Get settings
        settings = get_settings("config.yaml")
        
        # Create enhanced sheets manager
        print("Initializing Google Sheets manager...")
        sheets_manager = EnhancedGoogleSheetsManager(settings.google_sheets)
        
        # Initialize connection
        success = await sheets_manager.initialize()
        if not success:
            print("❌ Failed to initialize Google Sheets manager")
            return
        
        print("✅ Google Sheets manager initialized successfully!")
        
        # Create new sheet
        new_sheet_name = "Brand_Monitoring_New"
        print(f"Creating new sheet: {new_sheet_name}")
        
        try:
            # Create new worksheet
            loop = asyncio.get_event_loop()
            new_worksheet = await loop.run_in_executor(
                None,
                sheets_manager._spreadsheet.add_worksheet,
                new_sheet_name,
                1000,  # rows
                20     # columns (for Stage 2 enhanced data)
            )
            
            print(f"✅ Successfully created new sheet: {new_sheet_name}")
            print(f"Sheet ID: {new_worksheet.id}")
            
            # Update the configuration to use the new sheet
            print("\nUpdating configuration...")
            
            # Read current config
            with open("config.yaml", "r") as f:
                config_content = f.read()
            
            # Update worksheet name
            config_content = config_content.replace(
                "worksheet_name: \"Brand_Monitoring\"",
                f"worksheet_name: \"{new_sheet_name}\""
            )
            
            # Write updated config
            with open("config.yaml", "w") as f:
                f.write(config_content)
            
            print(f"✅ Configuration updated to use new sheet: {new_sheet_name}")
            
            # Test the new sheet
            print("\nTesting new sheet...")
            sheets_manager._worksheet = new_worksheet
            
            # Setup headers
            await sheets_manager._setup_enhanced_headers()
            print("✅ Enhanced headers setup completed")
            
            print(f"\n🎉 New monitoring sheet '{new_sheet_name}' created successfully!")
            print("The system will now use this new sheet for all future monitoring results.")
            
        except Exception as e:
            print(f"❌ Failed to create new sheet: {str(e)}")
            return
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

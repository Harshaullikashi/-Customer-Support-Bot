"""
Test script to verify installation and basic functionality
"""

import sys
import importlib
import os

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        "transformers",
        "sentence_transformers", 
        "torch",
        "PyPDF2",
        "numpy",
        "sklearn",
        "logging",
        "json",
        "datetime",
        "re",
        "random",
        "typing"
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError as e:
            print(f"[FAIL] {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n[FAIL] Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All packages imported successfully!")
        return True

def test_file_structure():
    """Test if required files exist."""
    required_files = [
        "support_bot_agent.py",
        "faq.txt",
        "requirements.txt",
        "README.md"
    ]
    
    print("\nTesting file structure...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n[FAIL] Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n[OK] All required files present!")
        return True

def test_basic_functionality():
    """Test basic bot functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from support_bot_agent import SupportBotAgent
        
        # Test initialization
        print("Initializing bot...")
        bot = SupportBotAgent("faq.txt")
        print("[OK] Bot initialized successfully")
        
        # Test simple query
        print("Testing simple query...")
        response = bot.answer_query("How do I reset my password?")
        print(f"[OK] Query response: {response[:100]}...")
        
        # Test statistics
        print("Testing statistics...")
        stats = bot.get_statistics()
        print(f"[OK] Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Customer Support Bot - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! The bot is ready to use.")
        print("Run 'python support_bot_agent.py' to start the bot.")
    else:
        print("\n[WARNING] Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

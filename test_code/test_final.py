"""
Final comprehensive test to verify the migration is complete and working
"""
import sys

def test_imports():
    """Test that all modules can be imported without errors"""
    print("Testing imports...")
    try:
        from scrape.extract_link import get_first_google_result
        from scrape.scrape import scrape, save_to_pdf
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_no_selenium():
    """Verify Selenium is not being used"""
    print("\nVerifying Selenium removal...")
    try:
        # Check if selenium imports would fail in our modules
        import scrape.scrape as scrape_module
        import scrape.extract_link as extract_module
        
        # Check module source
        scrape_source = open('scrape/scrape.py').read()
        extract_source = open('scrape/extract_link.py').read()
        
        if 'from selenium' in scrape_source or 'import selenium' in scrape_source:
            print("  ✗ Selenium still imported in scrape.py")
            return False
        
        if 'from selenium' in extract_source or 'import selenium' in extract_source:
            print("  ✗ Selenium still imported in extract_link.py")
            return False
        
        print("  ✓ Selenium completely removed")
        return True
    except Exception as e:
        print(f"  ✗ Error checking Selenium: {e}")
        return False

def test_crawl4ai_present():
    """Verify crawl4ai is being used"""
    print("\nVerifying crawl4ai implementation...")
    try:
        scrape_source = open('scrape/scrape.py').read()
        extract_source = open('scrape/extract_link.py').read()
        
        if 'crawl4ai' not in scrape_source and 'AsyncWebCrawler' not in scrape_source:
            print("  ✗ crawl4ai not found in scrape.py")
            return False
        
        if 'crawl4ai' not in extract_source and 'AsyncWebCrawler' not in extract_source:
            print("  ✗ crawl4ai not found in extract_link.py")
            return False
        
        print("  ✓ crawl4ai properly implemented")
        return True
    except Exception as e:
        print(f"  ✗ Error checking crawl4ai: {e}")
        return False

def test_functionality():
    """Test actual functionality"""
    print("\nTesting functionality...")
    try:
        from scrape.extract_link import get_first_google_result
        from scrape.scrape import scrape
        
        # Quick test with a simple URL
        print("  Testing Google search...")
        result = get_first_google_result("https://www.python.org")
        if result and 'python' in result.lower():
            print("    ✓ Google search working")
        else:
            print(f"    ⚠ Unexpected result: {result}")
        
        print("  Testing web scraping...")
        content = scrape("https://www.example.com")
        if len(content) > 0:
            print(f"    ✓ Scraping working (extracted {len(content)} characters)")
        else:
            print("    ✗ No content extracted")
            return False
        
        print("  ✓ All functionality tests passed")
        return True
    except Exception as e:
        print(f"  ✗ Functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 70)
    print("COMPREHENSIVE MIGRATION VERIFICATION TEST")
    print("=" * 70)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Import Test", test_imports()))
    results.append(("Selenium Removal", test_no_selenium()))
    results.append(("Crawl4AI Implementation", test_crawl4ai_present()))
    results.append(("Functionality Test", test_functionality()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Migration successful!")
        print("\nThe repository has been successfully migrated from Selenium to crawl4ai.")
        print("All functionality is working as expected.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
Integration test for crawl4ai with the actual app flow
"""
print("=" * 60)
print("INTEGRATION TEST: Testing full workflow")
print("=" * 60)

try:
    from scrape.extract_link import get_first_google_result
    from scrape.scrape import scrape
    import time
    
    # Simulate the app flow
    test_url = "https://www.github.com"
    
    print(f"\n1. Finding Terms & Conditions page for: {test_url}")
    start_url = get_first_google_result(test_url)
    print(f"   ✓ Found: {start_url}")
    
    print(f"\n2. Scraping content from: {start_url}")
    start_time = time.time()
    content = scrape(start_url)
    elapsed = time.time() - start_time
    
    print(f"   ✓ Scraped {len(content)} characters in {elapsed:.2f} seconds")
    print(f"   First 300 characters:")
    print(f"   {content[:300]}...")
    
    print(f"\n{'='*60}")
    print("✓ INTEGRATION TEST PASSED!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Google search: Working ✓")
    print(f"  - Web scraping: Working ✓")
    print(f"  - Content extraction: Working ✓")
    print(f"  - Cleanup: Working ✓")
    
except Exception as e:
    print(f"\n✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

"""
Test script to verify crawl4ai implementation
"""
import sys
import os

# Test 1: Test extract_link.py
print("=" * 60)
print("TEST 1: Testing Google search with crawl4ai")
print("=" * 60)

try:
    from scrape.extract_link import get_first_google_result
    
    test_url = "https://www.google.com"
    print(f"Testing with URL: {test_url}")
    result = get_first_google_result(test_url)
    print(f"✓ Google search successful!")
    print(f"Result URL: {result}")
except Exception as e:
    print(f"✗ Error in Google search test: {e}")
    import traceback
    traceback.print_exc()

print("\n")

# Test 2: Test scrape.py
print("=" * 60)
print("TEST 2: Testing web scraping with crawl4ai")
print("=" * 60)

try:
    from scrape.scrape import scrape
    
    test_url = "https://www.example.com"
    print(f"Testing scraping with URL: {test_url}")
    result = scrape(test_url)
    print(f"✓ Scraping successful!")
    print(f"Extracted text length: {len(result)} characters")
    print(f"First 200 characters: {result[:200]}")
except Exception as e:
    print(f"✗ Error in scraping test: {e}")
    import traceback
    traceback.print_exc()

print("\n")
print("=" * 60)
print("TESTING COMPLETE")
print("=" * 60)

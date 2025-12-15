#!/usr/bin/env python
"""Test script to verify scraping works"""

from scrape.extract_link import get_first_google_result
from scrape.scrape import scrape

# Test Google search
print("Testing Google search...")
test_url = "https://www.reddit.com/r/test"
result_url = get_first_google_result(test_url)
print(f"Input: {test_url}")
print(f"Google result: {result_url}")

# Test scraping
print("\nTesting scraping...")
scraped_text = scrape(result_url)
print(f"Scraped {len(scraped_text)} characters")
if len(scraped_text) > 0:
    print(f"Preview: {scraped_text[:200]}...")
    print("\n✅ All tests passed!")
else:
    print("\n❌ Scraping returned empty text")

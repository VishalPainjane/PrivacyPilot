"""
Simple test to verify scraper is working
"""
from scrape.scrape import scrape
from scrape.extract_link import get_first_google_result

print("=" * 60)
print("Testing Google Search")
print("=" * 60)

# Test 1: Google search
query = "reddit privacy policy"
print(f"\nSearching for: {query}")
url = get_first_google_result(query)
print(f"Result: {url}")

print("\n" + "=" * 60)
print("Testing Scraper")
print("=" * 60)

# Test 2: Scrape a simple URL
test_url = "https://www.reddit.com/policies/privacy-policy"
print(f"\nScraping: {test_url}")
result = scrape(test_url)

print(f"\nTitle: {result.get('title', 'N/A')}")
print(f"URL: {result.get('url', 'N/A')}")
print(f"Text length: {len(result.get('text', ''))} characters")
print(f"First 500 chars:\n{result.get('text', '')[:500]}")

print("\n" + "=" * 60)
print("✅ Test Complete!")
print("=" * 60)

# Crawl4AI Quick Reference

## What Changed?

The PrivacyPilot project now uses **crawl4ai** instead of Selenium for web scraping. This means:

- ✅ No need to install browser drivers (ChromeDriver, GeckoDriver)
- ✅ Faster async operations
- ✅ Cleaner, simpler code
- ✅ Automatic resource cleanup

## Installation

```bash
pip install -r requirements.txt
```

That's it! Crawl4ai will automatically handle browser setup on first run.

## Usage

### Google Search for Terms & Conditions

```python
from scrape.extract_link import get_first_google_result

url = "https://www.example.com"
terms_url = get_first_google_result(url)
# Returns: URL of the terms and conditions page
```

### Scrape Web Content

```python
from scrape.scrape import scrape

url = "https://www.example.com/terms"
content = scrape(url)
# Returns: Text content from the page and linked pages with keywords
```

### Save to PDF

```python
from scrape.scrape import save_to_pdf

text = "Your extracted content here"
save_to_pdf(text, "output/terms.pdf")
```

## Complete Example

```python
from scrape.extract_link import get_first_google_result
from scrape.scrape import scrape, save_to_pdf

# 1. Find terms page
url = "https://www.github.com"
terms_url = get_first_google_result(url)
print(f"Found: {terms_url}")

# 2. Scrape content
content = scrape(terms_url)
print(f"Extracted {len(content)} characters")

# 3. Save to PDF
save_to_pdf(content, "data/terms_and_policies.pdf")
```

## Testing

Run the test files to verify everything is working:

```bash
# Basic functionality test
python test_crawl4ai.py

# Integration test
python test_integration.py

# Comprehensive verification
python test_final.py
```

## Troubleshooting

### First Run Takes Long

- Normal! Crawl4ai downloads browser binaries on first run
- Subsequent runs will be much faster

### Connection Errors

- The tool has automatic fallback mechanisms
- If crawl4ai fails, it falls back to using requests library
- Most errors are non-fatal and automatically handled

### Verbose Logging

- Logging is set to minimal by default
- See the `[INIT]`, `[FETCH]`, `[SCRAPE]` messages during operation
- These are informational and don't indicate errors

## Performance Notes

- Average scraping time: 2-10 seconds per page
- Google search: 1-3 seconds
- Respects the 7-second timeout for recursive scraping
- Automatically limits content to 200,000 characters

## Key Features Preserved

✅ Keyword-based link discovery  
✅ Recursive scraping within same domain  
✅ 7-second timeout for scraping operations  
✅ Content cleaning and formatting  
✅ PDF generation with Unicode support

## Differences from Selenium

| Feature         | Selenium         | Crawl4AI  |
| --------------- | ---------------- | --------- |
| Driver Setup    | Manual           | Automatic |
| Performance     | Slower           | Faster    |
| Resource Usage  | Higher           | Lower     |
| Code Complexity | More boilerplate | Cleaner   |
| Async Support   | Limited          | Native    |
| Cleanup         | Manual           | Automatic |

## Support

For issues or questions:

1. Check the test files for examples
2. Review [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) for details
3. See crawl4ai documentation: https://crawl4ai.com/

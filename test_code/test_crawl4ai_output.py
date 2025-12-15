import asyncio
from crawl4ai import AsyncWebCrawler

async def test():
    crawler = AsyncWebCrawler(verbose=True)
    result = await crawler.arun(url='https://www.example.com')
    
    print("\n=== Result Object ===")
    print(f"Type: {type(result)}")
    print(f"\nAttributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
    
    # Access the actual result
    if hasattr(result, 'result'):
        actual_result = result.result
    else:
        actual_result = result
    
    if hasattr(actual_result, 'html'):
        print(f"\nHTML length: {len(actual_result.html)}")
        print(f"HTML preview: {actual_result.html[:200]}...")
    
    if hasattr(actual_result, 'markdown'):
        print(f"\nMarkdown length: {len(actual_result.markdown)}")
        print(f"Markdown preview: {actual_result.markdown[:200]}...")
        
    if hasattr(actual_result, 'text'):
        print(f"\nText length: {len(actual_result.text)}")
        print(f"Text preview: {actual_result.text[:200]}...")
    
    await crawler.close()

asyncio.run(test())

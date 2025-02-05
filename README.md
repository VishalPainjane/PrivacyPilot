# Privacy-Pilot

## Overview
A Python-based tool for analyzing, and evaluating website Terms & Conditions documents with a focus on privacy and data protection compliance.

## Features
- GDPR compliance scoring
- Privacy and legal compliance analysis
- Web scraping of Terms & Conditions pages
- Multi-query document retrieval
- Comprehensive JSON output of privacy metrics

## Prerequisites
- Python 3.8+
- Google Chrome
- ChromeDriver

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/privacy-analyzer.git
cd privacy-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file
- Add your Groq API key: `GROQ_API_KEY=your_api_key_here`

## Configuration

### Model Selection
Modify the `llm` initialization in `get_json()` to choose different language models:
- `mixtral-8x7b-32768`
- `llama3-8b-8192`
- Other Groq-supported models

### Embedding Model
Update `local_model_path` to use a different Hugging Face embedding model.

## Usage

```python
from main import get_json

# Analyze Terms & Conditions
url = "https://example.com/terms"
results = get_json(url)
```

## Dependencies
- LangChain
- Selenium
- Hugging Face Transformers
- Groq API
- Chroma Vector Store

## Limitations
- Requires Chrome WebDriver
- Accuracy depends on webpage structure
- Limited to web-scraping capabilities

## Disclaimer
This tool provides an automated analysis and should not replace professional legal advice.

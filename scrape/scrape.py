import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options

from fpdf import FPDF



start_time = time.time()

# keywords = ['privacy', 'terms', 'condition', 'legal']

keywords = [
    'term', 'terms', 'condition', 'conditions', 'policy', 'policies',
    'agreement', 'agreements', 'statement', 'statements',
    'protection', 'protections', 'clause', 'clauses', 'disclaimer', 'disclaimers', 'legal',
    'privacy', 'data', 'service', 'user', 'cookie', 'third-party', 'security', 'consent',
    'notice', 'regulation', 'compliance', 'contract', 'contracts', 'rights', 'sharing',
    'opt-out', 'opt-in', 'disclosure', 'retention', 'encryption', 'Terms of service', 'safety',
    'help', 'support', 'contact', 'information', 'cookies', 'cookies-policy', 
    'user agreement', 'cookie policy', 'terms of service', 'privacy policy', 'privacy-policy'
    'privacy statement', 'data protection', 'legal terms', 'legal agreement',
    'service terms', 'service agreement', 'terms and conditions', 't&c', 't and c',
    't n c', 't&cs', 't&cs policy', 'tnc policy', 'tncs','Privacy policy',

    'regulation', 'compliance', 'contract', 'contracts', 'user consent',
    'data usage', 'data collection', 'user rights', 'third-party sharing',
    'opt-out', 'opt-in', 'disclosure', 'retention', 'security policy',
]


visited = set()
extracted_text = ""

options = Options()

options.add_argument("--headless") 
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

# driver = webdriver.Chrome(options=options)
driver = webdriver.Firefox(options=options)

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def contains_keyword(url):
    return any(keyword in url.lower() for keyword in keywords)

count = 0
flag = 0
start_link = ""



def scrape(url):
    global start_time
    global extracted_text
    global count
    global start_link
    global flag

    if count == 0:
        start_link = url
        count += 1

    if time.time() - start_time > 7:
        return extracted_text[:200000] or ""

    if url in visited:
        return

    visited.add(url)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    text = soup.get_text()

    if extracted_text is None:
        extracted_text = ""
    extracted_text += text

    for link in soup.find_all('a', href=True):
        if time.time() - start_time > 7:
            return extracted_text[:200000] or ""

        domain = start_link[len("https://"):].split(".")[0]
        if domain not in link['href']:
            continue

        full_url = urljoin(url, link['href'])
        if flag != 0:
            if full_url in visited:
                continue

        if is_valid(full_url) and contains_keyword(full_url):
            result = scrape(full_url)
            flag = 1
            if result:
                extracted_text += result

    return extracted_text[:200000] or ""


from fpdf import FPDF

def save_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()

    font_path = r"scrape\DejaVuSans.ttf"
    try:
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font('DejaVu', size=12)
    except RuntimeError as e:
        print(f"Error loading font: {e}")
        return

    count = 0
    for line in text.split('\n'):
        print(count)
        try:
            safe_line = ''.join(c if ord(c) < 65536 else '?' for c in line)
            pdf.multi_cell(0, 10, safe_line)
            count+=1
        except Exception as e:
            print(f"Error writing line: {line}\n{e}")

    try:
        pdf.output(filename)
        print(f"PDF saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving PDF: {e}")



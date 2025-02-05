import time

# Start the timer
start_time = time.time()




import warnings
from pprint import pprint

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from scrape.extract_link import get_first_google_result

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# from scrape import start_url
import re

import json
import dotenv
import os

warnings.filterwarnings("ignore")

dotenv.load_dotenv()

# Get the API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")





# url = "https://www.youtube.com/watch?v=YAgIh4aFawU&ab_channel=AIExplained"
# url = "https://open.spotify.com/genre/0JQ5DArNBzkmxXHCqFLx2J"
url = "https://www.reddit.com/r/Minecraft/comments/1hjza9r/ytber_stole_my_build_without_credit_and_hides/"

# Get first standard Google result
start_url = get_first_google_result(url)
print("First Google result link:", start_url)


from scrape.scrape import save_to_pdf, scrape
from scrape.scrape import driver


a = scrape(start_url)
driver.quit()

def clean_and_compact(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    
    # Convert multiple newlines to single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s([.,!?:;&()])', r'\1', text)
    text = re.sub(r'([.,!?:;&()])\s', r'\1 ', text)
    
    # Remove extra whitespace around hyphens
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Fix spacing around quotes
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    # Fix multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove leading/trailing whitespace
    return text.strip()

cleaned_text = clean_and_compact(a)
print((len(cleaned_text)))

print("Saving the extracted text to a PDF file...")
save_to_pdf(cleaned_text, r"data/terms_and_policies.pdf")




print("LLM is initializing...")
# Initialize the language model
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    # model="llama3-8b-8192",
    # model="llama-3.2-11b-vision-preview",
    # model="llama-3.2-90b-vision-preview",
    # model="llama-3.3-70b-versatile",
    # model="gemma2-9b-it",
    temperature=0.2,
)
print("LLM initialized successfully.")

# Load the PDF document
path = r"data\terms_and_policies.pdf"

print("Loading the PDF document...")
# Load the documents from the PDF
loader = PyPDFLoader(path)
docs = loader.load()
print("PDF document loaded successfully.")


print("Splitting the documents into smaller chunks...")
# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
print("Documents split successfully.")


print("Creating the vector store...")
# Create the vector store
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
db = Chroma.from_documents(documents=documents, embedding=embeddings)
print("Vector store created successfully.")

# Define the output parser
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) 


# Define the prompt template
output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. Provide these alternative questions separated by newlines.
Original question: {question}""",
)

# Define the retrieval chain
llm_chain = QUERY_PROMPT | llm | output_parser

# Create the retriever
print("Creating the retriever...")
retriever = MultiQueryRetriever(
    retriever=db.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)
print("Retriever created successfully.")

print("Invoking the retriever...")
# Add the documents to the retriever
unique_docs = retriever.invoke('''Extract chunks of text from the Terms & Conditions document that are directly relevant to the following parameters for further analysis. 
                                Each chunk should provide full context and clear information about the respective category.
                               
                                Provide most exclusive and relevant information for each parameter.

                                Parameters to Focus On:

                                    Account Control
                                    Data Collection
                                    Data Deletion
                                    Data Sharing
                                    Legal Rights
                                    Privacy Controls
                                    Security Measures
                                    Terms Changes
                                    Transparency
                                    User Content Rights'''
                               )
                               
print(len(unique_docs))


SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
Answer only contains a JSON object with scores and quotes as per the given format:

<context>
{context}
</context>
"""

# Define the question answering prompt
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the document chain
document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

# Invoke the document chain
print("Invoking the document chain...")
response = document_chain.invoke(
    {
        "context": unique_docs[:5],
        "messages": [
            HumanMessage(
                content='''
                    Analyze this Terms & Conditions text gathered by scraping the web pages, and return a single JSON object containing only scores and direct full-sentance quotes. 
                    
                    Provide most exclusive and relevant information according to the users perspective, as users are concerned about their privacy and data security.

                    Structure of JSON object is important.
                    
                    Return your analysis in this exact format:
                    {
                        "scores": {
                            "account_control": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "data_collection": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "data_deletion": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "data_sharing": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "legal_rights": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "privacy_controls": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "security_measures": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "terms_changes": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "transparency": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            },
                            "user_content_rights": {
                                "quotes": [
                                    "exact quote 1",
                                    "exact quote 2"
                                ],
                                "score": 1-5
                            }
                        }, 
                        "metadata": {
                            "risk_percentage": 0-100,
                            "risk_level": "Very High Risk|High Risk|Moderate Risk|Low Risk",
                            "GDPR_compliance_score": 0-5,
                            "additional_notes": "Detailed observations about GDPR compliance and related strengths or shortcomings."
                        }
                    }
                    
                    Score each parameter based on these criteria:
                        Lawfulness, Fairness, and Transparency
                        Purpose Limitation
                        Data Minimization
                        Accuracy
                        Storage Limitation
                        Integrity and Confidentiality (Security)
                        Accountability
                        Data Subject Rights
                        Consent Management
                        Data Portability


                    Important rules for quotes:

                    Use full-sentences for quotes
                    Include exact text from the document
                    Focus on the most concerning/relevant parts
                    Keep quotes concise but complete
                    Include context when needed
                    If a quote shows positive aspects, include it too
                    Quotes should directly support the score given

                    QUOTES ARE MANDATORY FOR EACH PARAMETER, REPRESENTED ACCURATELY AND CONCISELY, FROM THE DOCUMENT.

                    Avoid risk_percentage to be multiples of 10 from the result. 

                    Risk Level Calculation:

                    Critical Concern: <30%
                    Elevated Concern: 30-44%
                    Moderate Concern: 45-50%
                    Caution Advised: 51-55%
                    Generally Favorable: 56-70%
                    Highly Favorable: >70%

                    get the score the with the information provided in the document and provide the risk level and GDPR compliance score based on the score.

                    '''
            ),
        ],
    }
)

pprint(response)

# Extract the JSON object from the response
def extract_json(a):
    start_index = a.find('{') 
    end_index = a.rfind('}')

    if start_index != -1 and end_index != -1:
        json_part = a[start_index:end_index + 1]
        with open('output.json', 'w') as json_file:
            json_file.write(json_part)
    else:
        print("No JSON found in the response.")

extract_json(response)


end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Time required to run the code: {elapsed_time:.2f} seconds")

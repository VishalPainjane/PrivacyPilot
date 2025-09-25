import time
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
import re
import json
import dotenv
import os

warnings.filterwarnings("ignore")
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_json(url: str, progress_callback=None) -> dict:
    if progress_callback:
        progress_callback(2)

    start_url = get_first_google_result(url)
    print("First Google result link:", start_url)

    if progress_callback:
        progress_callback(8)

    from scrape.scrape import save_to_pdf, scrape
    from scrape.scrape import driver

    if progress_callback:
        progress_callback(12)

    a = scrape(start_url)
    driver.quit()

    def clean_and_compact(text):
        text = re.sub(r'[^\x20-\x7E]+', ' ', text) 
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'\s([.,!?&()])', r'\1', text)  
        text = re.sub(r'([.,!?&()])\s', r'\1 ', text)  
        return text.strip()  

    cleaned_text = clean_and_compact(a)
    print((len(cleaned_text)))
    
    if progress_callback:
        progress_callback(18)

    print("Saving the extracted text to a PDF file...")
    save_to_pdf(cleaned_text, r"data/terms_and_policies.pdf")

    if progress_callback:
        progress_callback(30)

    print("LLM is initializing...")
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.2,
    )
    print("LLM initialized successfully.")

    path = r"data\terms_and_policies.pdf"

    print("Loading the PDF document...")
    
    if progress_callback:
        progress_callback(42)
    
    loader = PyPDFLoader(path)
    docs = loader.load()
    print("PDF document loaded successfully.")

    print("Splitting the documents into smaller chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)
    print("Documents split successfully.")

    if progress_callback:
        progress_callback(48)

    print("Creating the vector store...")
    local_model_path = r"model"
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

    db = Chroma.from_documents(documents=documents, embedding=embeddings)
    print("Vector store created successfully.")
    
    if progress_callback:
        progress_callback(65)

    class LineListOutputParser(BaseOutputParser[List[str]]):
        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines)) 

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

    llm_chain = QUERY_PROMPT | llm | output_parser

    print("Creating the retriever...")
    retriever = MultiQueryRetriever(
        retriever=db.as_retriever(search_kwags = {"k":3}), llm_chain=llm_chain, parser_key="lines"
    )
    
    print("Retriever created successfully.")
    if progress_callback:
        progress_callback(69)

    print("Invoking the retriever...")
    unique_docs = retriever.invoke('''Generate a structured daily schedule based on the physical information and diet plan of a person.

            Each schedule should be well-organized, ensuring that meal timings, workouts, hydration, and rest periods align with the individual's fitness goals and nutritional needs.

            Provide the most relevant and optimized routine for each of the following aspects:

            Wake-up Routine

            Morning Hydration & Nutrition

            Exercise & Workout Plan

            Breakfast

            Mid-Morning Snack

            Lunch

            Afternoon Activity & Hydration

            Evening Snack

            Dinner

            Night Routine & Recovery

            Ensure the schedule maintains a balance between energy intake, physical activity, and rest for overall well-being.'''
                                )
                                
    print(len(unique_docs))

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    Answer only contains a JSON object with scores and quotes as per the given format:

    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    if progress_callback:
        progress_callback(85)

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
                        }'''
                ),
            ],
        }
    )

    pprint(response)

    def extract_json(a):
        start_index = a.find('{') 
        end_index = a.rfind('}')

        if start_index != -1 and end_index != -1:
            json_part = a[start_index:end_index + 1]
            with open('output.json', 'w') as json_file:
                json_file.write(json_part)
        else:
            print("No JSON found in the response.")
        return json_part

    extract_json(response)

    with open('output.json', 'r') as file:
        data = json.load(file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time required to run the code: {elapsed_time:.2f} seconds")

    if progress_callback:
        progress_callback(100)

    return data

json_ob = get_json("https://www.reddit.com/r/Minecraft/comments/1hjza9r/ytber_stole_my_build_without_credit_and_hides/")
print("Done")
print(json_ob)

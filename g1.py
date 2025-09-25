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



def get_json(path: str, progress_callback=None) -> dict:

#     if progress_callback:
#         progress_callback(2)

#     # Get first standard Google result
#     start_url = get_first_google_result(url)
#     print("First Google result link:", start_url)

#     if progress_callback:
#         progress_callback(8)

#     from scrape.scrape import save_to_pdf, scrape
#     from scrape.scrape import driver

#     if progress_callback:
#         progress_callback(12)

#     a = scrape(start_url)
#     driver.quit()

#     def clean_and_compact(text):
#         text = re.sub(r'[^\x20-\x7E]+', ' ', text) 
#         text = re.sub(r'\s+', ' ', text)  
#         text = re.sub(r'\s([.,!?&()])', r'\1', text)  
#         text = re.sub(r'([.,!?&()])\s', r'\1 ', text)  
#         return text.strip()  

#     cleaned_text = clean_and_compact(a)
#     print((len(cleaned_text)))
    
#     if progress_callback:
#         progress_callback(18)

#     print("Saving the extracted text to a PDF file...")
#     save_to_pdf(cleaned_text, r"data/terms_and_policies.pdf")



    if progress_callback:
        progress_callback(30)

    print("LLM is initializing...")
    # Initialize the language model
    llm = ChatGroq(
        # model="mixtral-8x7b-32768",
        model="llama3-8b-8192",
        # model="llama-3.2-11b-vision-preview",
        # model="llama-3.2-90b-vision-preview",
        # model="llama-3.3-70b-versatile",
        # model="gemma2-9b-it",
        temperature=0.2,
    )
    print("LLM initialized successfully.")

    # Load the PDF document
    # path = r"t2.pdf"

    print("Loading the PDF document...")
    
    if progress_callback:
        progress_callback(42)
    
    
    loader = PyPDFLoader(path)
    docs = loader.load()
    print("PDF document loaded successfully.")


    print("Splitting the documents into smaller chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
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
    # retriever = MultiQueryRetriever(
    #     retriever=db.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    # )


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
                        
                        Structure of JSON object is important.
                        
                        Return your analysis in this exact format:
                        {
                        "assessment": {
                            "physical": {
                            "strengths": ["Familiar with compound movements", "Knowledge of strength training"],
                            "limitations": ["BMI of 16.83 (underweight)", "Limited energy reserves"],
                            "fitness_level": "Intermediate"
                            },
                            "nutrition": {
                            "current": "Insufficient calories with reasonable food choices",
                            "recommended": "50% carbs / 25% protein / 25% fat with caloric surplus",
                            "improvements": ["Increase calories", "Add calorie-dense foods", "More complex carbs"]
                            },
                            "academic": {
                            "peak_times": ["Morning lectures", "Afternoon study", "Evening exam prep"],
                            "workout_windows": ["Early morning (6-8am)", "Late afternoon (4-6pm)"]
                            }
                        },
                        "schedule": {
                            "class_day": [
                            {
                                "time": "06:30 AM",
                                "type": "Meal",
                                "name": "Calorie-Dense Breakfast",
                                "details": "High-protein, high-calorie breakfast"
                            },
                            {
                                "time": "10:00 AM",
                                "type": "Meal",
                                "name": "Mid-Morning Snack",
                                "details": "Quick calorie-dense snack between classes"
                            },
                            {
                                "time": "12:30 PM",
                                "type": "Meal",
                                "name": "Substantial Lunch",
                                "details": "Focus on protein, vegetables, and complex carbs"
                            },
                            {
                                "time": "04:00 PM",
                                "type": "Workout",
                                "name": "Strength Training",
                                "details": "Compound movements for muscle growth"
                            },
                            {
                                "time": "05:00 PM",
                                "type": "Meal",
                                "name": "Post-Workout Nutrition",
                                "details": "Protein and carbs for recovery"
                            },
                            {
                                "time": "07:30 PM",
                                "type": "Meal",
                                "name": "Nutrient-Dense Dinner",
                                "details": "Complete nutrition and adequate calories"
                            },
                            {
                                "time": "09:30 PM",
                                "type": "Meal",
                                "name": "Evening Protein Snack",
                                "details": "Slow-digesting protein before bed"
                            }
                            ],
                            "exam_day": [
                            {
                                "time": "06:00 AM",
                                "type": "Meal",
                                "name": "Brain-Boosting Breakfast",
                                "details": "Omega-3s and complex carbs for mental energy"
                            },
                            {
                                "time": "07:00 AM",
                                "type": "Workout",
                                "name": "Light Morning Movement",
                                "details": "Brief, low-intensity exercise for mental alertness"
                            },
                            {
                                "time": "Every 2-3 hours",
                                "type": "Meal",
                                "name": "Brain-Fueling Snacks",
                                "details": "Regular small meals to maintain cognitive function"
                            }
                            ]
                        },
                        "adaptation": {
                            "low_energy": [
                            "Reduce workout intensity, maintain protein intake",
                            "Add digestible carbs before high-concentration classes",
                            "Increase liquid nutrition (smoothies, protein shakes)"
                            ],
                            "time_constraints": [
                            "20-30 minute compound movement workouts",
                            "Prepare portable, calorie-dense meals in advance",
                            "5-10 minute movement breaks between study sessions"
                            ],
                            "exam_periods": [
                            "Reduce workout intensity, maintain frequency",
                            "Increase omega-3s and antioxidants",
                            "Regular small meals during study sessions"
                            ]
                        },
                        "semester_plan": {
                            "early_weeks_1-4": {
                            "fitness": "Establish 3-4 days/week strength training",
                            "nutrition": "5-6 meals daily with increasing portions",
                            "goals": "0.5-1 lb gain per week"
                            },
                            "mid_weeks_5-10": {
                            "fitness": "Progressive overload on compound lifts",
                            "nutrition": "Optimize pre/post workout nutrition",
                            "goals": "Consistent weight gain and strength progression"
                            },
                            "finals_weeks_11-15": {
                            "fitness": "Maintenance with reduced volume",
                            "nutrition": "Focus on cognitive-enhancing nutrients",
                            "goals": "Maintain gains while managing stress"
                            }
                        },
                        "implementation": {
                            "habits": [
                            "Meal prep twice weekly",
                            "Consistent workout schedule",
                            "Keep healthy snacks accessible",
                            "Workout partner for accountability"
                            ],
                            "minimum_protocol": {
                            "time": "30 minutes/day",
                            "essentials": [
                                "3 strength sessions weekly",
                                "1.6g protein per kg body weight",
                                "5-6 meals daily",
                                "8 hours sleep priority"
                            ]
                            }
                        }
                        }


                        Important rules for quotes:

                        Use full-sentences for quotes
                        Include exact text from the document
                        Focus on the most concerning/relevant parts
                        Keep quotes concise but complete
                        Include context when needed
                        If a quote shows positive aspects, include it too
                        Quotes should directly support the score given

                        QUOTES ARE MANDATORY FOR EACH PARAMETER, REPRESENTED ACCURATELY AND CONCISELY, FROM THE DOCUMENT.

                        '''
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

json_ob = get_json("p1.pdf")
print("Done")
print(json_ob)
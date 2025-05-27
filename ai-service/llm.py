from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()  


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)


# Main Function
def generate_blog(prompt) -> str:
    prompt = prompt
    response = llm([HumanMessage(content=prompt)])
    return response.content

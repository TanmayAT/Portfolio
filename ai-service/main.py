from llm import generate_blog
from fallbackmodel import FallbackModel
from prompt import prompt_by_id
from fastapi import FastAPI



app = FastAPI()



@app.post("/genrate_content")
def genrate_content(query:str) : 
    
    try : 
        
        
        generate_blog(query) 
        
        
    except Exception as e : 
        
        
        FallbackModel.call_fallback(query)
        
        
    

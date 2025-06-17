from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from agent.quality_agent import QualityManagementAgent
import uvicorn
from typing import Optional
import os

app = FastAPI(
    title="AI Quality Companion API",
    description="API for quality management tools powered by AI",
    version="1.0.0"
)

# LLM Configuration
class LLMConfig(BaseModel):
    model: str = "groq:llama3-8b-8192"
    api_key: Optional[str] = None

# Global agent instance
agent = None

def get_agent(config: LLMConfig = Depends()) -> QualityManagementAgent:
    """
    Get or create the QualityManagementAgent instance with the specified configuration.
    """
    global agent
    if agent is None or agent.model != config.model:
        # Use API key from config or environment variable
        api_key = config.api_key or os.getenv(f"{config.model.split(':')[0].upper()}_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=f"API key not provided for {config.model}. Please provide it in the request or set the {config.model.split(':')[0].upper()}_API_KEY environment variable."
            )
        agent = QualityManagementAgent(model=config.model, api_key=api_key)
    return agent

class QualityRequest(BaseModel):
    prompt: str
    thread_id: str = "default_thread"
    llm_config: Optional[LLMConfig] = None

@app.post("/analyze")
async def analyze_quality(request: QualityRequest):
    """
    Analyze quality issues using the AI agent.
    
    Parameters:
    - prompt: The analysis request (e.g., FMEA, SIPOC, Fishbone diagram)
    - thread_id: Optional thread ID for conversation tracking
    - llm_config: Optional LLM configuration (model and API key)
    
    Returns:
    - Analysis results and generated file paths
    """
    try:
        # Get agent with configuration
        current_agent = get_agent(request.llm_config or LLMConfig())
        
        # Run the agent with the provided prompt
        response = current_agent.run(content=request.prompt, thread_id=request.thread_id)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint returning API information
    """
    return {
        "name": "AI Quality Companion API",
        "version": "1.0.0",
        "description": "API for quality management tools powered by AI",
        "endpoints": {
            "/analyze": "POST - Run quality analysis with AI agent",
            "/": "GET - API information"
        },
        "supported_models": {
            "groq": ["llama3-8b-8192"],
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet"]
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
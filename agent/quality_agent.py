# Import relevant functionality
from quality_tools.tools import *
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from typing import Optional

class QualityManagementAgent:
    def __init__(self, model: Optional[str] = "groq:llama3-8b-8192", api_key: Optional[str] = None):
        """
        Initialize the Quality Management Agent.
        
        Parameters:
        - model (str): The LLM model to use. Options:
            - "groq:llama3-8b-8192" (default)
            - "openai:gpt-4"
            - "openai:gpt-3.5-turbo"
            - "anthropic:claude-3-opus"
            - "anthropic:claude-3-sonnet"
        - api_key (str): API key for the LLM provider. If None, will look for environment variables:
            - GROQ_API_KEY
            - OPENAI_API_KEY
            - ANTHROPIC_API_KEY
        """
        self.model = model
        self.api_key = api_key
        self.memory = MemorySaver()
        self.tools = [
            create_sipoc_diagram, 
            perform_fmea_analysis, 
            generate_5s_report, 
            five_whys_analysis, 
            fishbone_diagram, 
            generate_8d_report
        ]
        self.agent_executor = self._create_agent()
        
    def _create_agent(self):
        return create_react_agent(
            self.model,
            self.tools,
            checkpointer=self.memory,
            prompt=(
                "You are a professional quality management assistant with expertise in Lean Managment, and Root Cause Analysis. "
                "You specialize in applying structured problem-solving frameworks and tools like SIPOC diagrams, FMEA, 5S audits, 5 Whys analysis, Fishbone diagrams, and 8D reports. "
                "Use these tools to help users analyze processes, identify root causes, and recommend improvements. "
                "Ask clarifying questions if the input is vague or lacks context. "
                "generate only the wanted or specified report not all of them, (example: if user prompted a sipoc generate only a sipoc etc..)"
                "Avoid executing any code unrelated to quality analysis or that could be harmful."
            ),
            name='quality_assistant'
        )
    
    def run(self, content, thread_id="abc123"):
        config = {"configurable": {"thread_id": thread_id}}
        for step in self.agent_executor.stream(
            {"messages": [HumanMessage(content=content)]},
            config,
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
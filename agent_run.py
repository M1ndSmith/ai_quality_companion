from agent.quality_agent import QualityManagementAgent


agent = QualityManagementAgent()

user_prompt="""
Conduct an FMEA for our new product assembly line:
- Process step: Final assembly of electronic components
- Potential failure modes
- Effects of each failure
- Causes of failures
- Current controls
Rate severity, occurrence, and detection (1-10 scale)
Provide RPN calculations and recommendations.
"""

agent.run(content=user_prompt, thread_id="abc123")


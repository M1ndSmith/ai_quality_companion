# AI Quality Companion

An ai assistant for quality management processes, combining LLM capabilities with lean management tools.

## Features

- Interactive AI agent 
- Specialized knowledge in:
  - Lean Manufacturing principles
  - Root Cause Analysis techniques
  - Vizualization tools

### Implemented Tools

1. **SIPOC Diagrams**
   - Supplier-Input-Process-Output-Customer visualization
   - Interactive diagram creation
   - Saves as PNG and CSV files

2. **FMEA Analysis**
   - Failure Mode Effects Analysis
   - Risk Priority Number (RPN) calculations
   - Automated recommendations
   - Generates risk matrix visualization

3. **5S Reporting**
   - Workplace organization assessment
   - Radar chart visualization
   - Scoring and recommendations

4. **5 Whys Analysis**
   - Root cause investigation
   - Visual chain analysis
   - Solution tracking

5. **Fishbone Diagrams**
   - Cause-effect analysis
   - Multi-category investigation
   - Interactive diagram generation
   - Saves as PNG file

6. **8D Reports**
   - Comprehensive problem-solving framework
   - Timeline visualization
   - Action tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/M1ndSmith/ai_quality_companion.git

# Navigate to project directory
cd ai_quality_companion

# Install required packages
pip install -r requirements.txt
```

## Usage

```python
from agent.quality_agent import QualityManagementAgent

# Initialize the agent
agent = QualityManagementAgent()

# Example: Create SIPOC diagram
agent.run(content="""
Create a SIPOC diagram for:
Process: Product Assembly
Suppliers: Raw Material Vendor, Component Supplier
Inputs: Raw Materials, Components, Assembly Instructions
Process Steps: Material Inspection, Assembly, Quality Check, Packaging
Outputs: Finished Products, Quality Reports
Customers: Distributors, End Users
""")

# Example: Perform FMEA analysis
agent.run(content="""
Conduct an FMEA for our new product assembly line:
- Process step: Final assembly of electronic components
- Potential failure modes
- Effects of each failure
- Causes of failures
- Current controls
Rate severity, occurrence, and detection (1-10 scale)
Provide RPN calculations and recommendations.
""")
```

## API Testing

A ready-to-use Jupyter notebook, `api_test_template.ipynb`, is included in this repository. It demonstrates how to interact with the FastAPI backend using different LLM providers (Groq, OpenAI, Anthropic) and various quality management prompts. You can use this notebook to quickly test the API and customize your own requests.

### Example: Testing with curl

You can also test the API directly using `curl` commands. Below are some examples for the `/analyze` endpoint:

#### FMEA with Groq (default)
```sh
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Conduct an FMEA for our new product assembly line: - Process step: Final assembly of electronic components - Potential failure modes - Effects of each failure - Causes of failures - Current controls Rate severity, occurrence, and detection (1-10 scale) Provide RPN calculations and recommendations.\", \"thread_id\": \"curl_test_1\"}"
```

#### SIPOC with OpenAI GPT-4
```sh
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Create a SIPOC diagram for: Process: Product Assembly Suppliers: Raw Material Vendor, Component Supplier Inputs: Raw Materials, Components, Assembly Instructions Process Steps: Material Inspection, Assembly, Quality Check, Packaging Outputs: Finished Products, Quality Reports Customers: Distributors, End Users\", \"thread_id\": \"curl_test_2\", \"llm_config\": {\"model\": \"openai:gpt-4\", \"api_key\": \"YOUR_OPENAI_API_KEY\"}}"
```

#### Fishbone with Anthropic Claude
```sh
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Create a fishbone diagram for quality issue: Problem: Inconsistent product quality Categories to analyze: - Machine factors - Method factors - Material factors - Measurement factors - Environment factors - People factors Show cause-effect relationships visually.\", \"thread_id\": \"curl_test_3\", \"llm_config\": {\"model\": \"anthropic:claude-3-sonnet\", \"api_key\": \"YOUR_ANTHROPIC_API_KEY\"}}"
```

#### Custom prompt
```sh
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Your custom prompt here\", \"thread_id\": \"curl_test_custom\", \"llm_config\": {\"model\": \"groq:llama3-8b-8192\", \"api_key\": \"YOUR_GROQ_API_KEY\"}}"
```

Replace `YOUR_OPENAI_API_KEY`, `YOUR_ANTHROPIC_API_KEY`, or `YOUR_GROQ_API_KEY` with your actual API keys as needed.

## Output Files

All generated files are saved in the current working directory:

- Diagrams are saved as PNG files (e.g., `fishbone_diagram.png`, `sipoc_diagram.png`)
- Data files are saved as CSV files (e.g., `sipoc_data.csv`, `fmea_table.csv`)
- Reports are saved as text files (e.g., `5s_report.txt`, `8d_report.txt`)

## Project Structure

```
ai_quality_companion/
├── agent/
│   ├── __init__.py
│   └── quality_agent.py
├── quality_tools/
│   ├── __init__.py
│   └── tools.py
├── __init__.py
├── agent_run.py
├── app.py
├── api_test_template.ipynb
├── prompts_examples.txt
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

- Python 3.8+
- LangGraph
- Groq API access (or you favourite llm)
- Matplotlib
- Additional dependencies in requirements.txt

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue in the GitHub repository.

## Acknowledgments

- Built with LangGraph and Groq (or any llm provider)
- Implements standard quality management methodologies
- Inspired by lean manufacturing tools

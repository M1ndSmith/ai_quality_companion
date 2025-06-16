# AI Quality Companion

An intelligent assistant for quality management processes, combining LLM capabilities with standard quality management tools.

## Features

- Interactive AI agent powered by LLama 3 (8-bit) via Groq
- Specialized knowledge in:
  - Six Sigma methodology
  - Lean Manufacturing principles
  - Root Cause Analysis techniques
  - Quality Management Systems

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
git clone https://github.com/yourusername/ai_quality_companion.git

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

## Output Files

All generated files are saved in the current working directory:

- Diagrams are saved as PNG files (e.g., `fishbone_diagram.png`, `sipoc_diagram.png`)
- Data files are saved as CSV files (e.g., `sipoc_data.csv`, `fmea_table.csv`)
- Reports are saved as text files (e.g., `5s_report.txt`, `8d_report.txt`)

## Project Structure

```
ai_quality_companion/
├── agent/
│   └── quality_agent.py
├── quality_tools/
│   ├── __init__.py
│   └── tools.py
├── agent_run.py
├── prompts_examples.txt
└── README.md
```

## Requirements

- Python 3.8+
- LangGraph
- Groq API access
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

- Built with LangGraph and Groq
- Implements standard quality management methodologies
- Inspired by Six Sigma
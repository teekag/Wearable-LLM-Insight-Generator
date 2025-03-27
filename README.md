# Wearable LLM Insight Generator

A modular AI system that ingests raw time-series data from wearables and generates personalized insights using Large Language Models (LLMs).

## Project Overview

This project aims to transform raw physiological data from wearable devices into actionable insights using LLMs. The system processes time-series data (glucose levels, insulin doses, heart rate variability, sleep metrics, activity data, etc.), extracts meaningful features, and generates personalized recommendations based on user goals and physiological patterns.

This system was built and validated on the publicly available DiaTrend dataset, containing real wearable data from individuals managing Type 1 diabetes. While the system is designed to be flexible for various wearable data types, it has been specifically optimized for glucose management and diabetes-related insights.

## Features

- **Data Ingestion**: Load and normalize data from various wearable devices (CSV, JSON formats)
- **Feature Engineering**: Extract meaningful features from raw physiological data
  - Glucose volatility and time-in-range metrics
  - Pre/post-meal glucose response patterns
  - Total daily insulin and dosing patterns
  - Comment sentiment and tag analysis
- **Prompt Engineering**: Generate structured prompts for LLMs based on physiological inputs
- **LLM Integration**: Generate insights using OpenAI, Mistral, or Gemini APIs
- **Conversational Agent**: Interactive coaching based on wearable data and user goals
- **Web Interface**: FastAPI backend with interactive dashboard

## Project Structure

```
wearable-llm-insight-generator/
├── data/
│   ├── raw/             # Raw data from wearable devices
│   │   └── diatrend/    # DiaTrend dataset files
│   └── processed/       # Processed features and insights
├── src/
│   ├── data_loader.py         # Data ingestion and normalization
│   ├── feature_engineer.py    # Feature extraction from wearable data
│   ├── insight_prompt_builder.py  # Prompt preparation for LLMs
│   ├── llm_engine.py          # LLM API integration
│   └── agent_simulator.py     # Conversational agent logic
├── notebooks/
│   ├── 01_diatrend_data_preparation.ipynb  # DiaTrend data loading and feature extraction
│   ├── 02_prompt_template_engine.ipynb     # Prompt engineering exploration
│   ├── 03_diatrend_llm_pipeline_demo.ipynb # End-to-end insight generation
│   └── 04_agent_interaction_simulation.ipynb # Agent conversation simulation
├── templates/           # HTML templates for web interface
├── static/              # Static files for web interface
├── outputs/             # Generated insights and conversations
├── app.py               # FastAPI web application
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (or Mistral/Gemini API keys for alternative LLM providers)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/wearable-llm-insight-generator.git
   cd wearable-llm-insight-generator
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```

5. Download the DiaTrend dataset:
   ```
   python -c "from src.data_loader import DataLoader; DataLoader().download_diatrend_dataset()"
   ```

### Running the Application

1. Start the FastAPI web server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Running the Notebooks

The notebooks demonstrate different aspects of the system:

1. **01_diatrend_data_preparation.ipynb**: Shows how to load, preprocess, and extract features from DiaTrend glucose and insulin data
2. **02_prompt_template_engine.ipynb**: Explores different prompt templates for generating insights
3. **03_diatrend_llm_pipeline_demo.ipynb**: Demonstrates the end-to-end pipeline for generating diabetes management insights
4. **04_agent_interaction_simulation.ipynb**: Shows how to use the agent for interactive conversations

To run a notebook:
```
jupyter notebook notebooks/01_diatrend_data_preparation.ipynb
```

## Usage Examples

### Loading and Processing DiaTrend Data

```python
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer

# Initialize components
data_loader = DataLoader()
feature_engineer = FeatureEngineer()

# Load DiaTrend data
diatrend_df = data_loader.load_diatrend_data()

# Segment by day
daily_data = data_loader.segment_diatrend_by_day(diatrend_df)

# Extract features
daily_features = {}
for day, df in daily_data.items():
    daily_features[day] = feature_engineer.extract_diatrend_features(df)
```

### Generating Diabetes Management Insights

```python
from src.insight_prompt_builder import InsightPromptBuilder
from src.llm_engine import LLMEngine

# Initialize components
prompt_builder = InsightPromptBuilder()
llm_engine = LLMEngine()

# Build prompt for diabetes management
prompt = prompt_builder.build_diatrend_prompt(
    features=daily_features["2023-01-01"],
    tone="coach",
    user_goal="Reduce post-meal glucose spikes"
)

# Generate insight
insight, metadata = llm_engine.generate_insight(prompt)
print(insight)
```

### Using the Conversational Agent

```python
from src.agent_simulator import AgentSimulator
from src.llm_engine import LLMEngine

# Initialize components
llm_engine = LLMEngine()
agent = AgentSimulator(llm_engine, daily_features["2023-01-01"])

# Set user goals
agent.set_user_goals({
    "glucose": "Reduce post-meal glucose spikes",
    "insulin": "Optimize insulin timing for better glucose control",
    "lifestyle": "Understand how exercise affects glucose levels"
})

# Chat with the agent
response, metadata = agent.process_message("Why are my glucose levels spiking after breakfast?")
print(response)
```

## Future Enhancements

- Integration with more wearable device APIs (Dexcom, Freestyle Libre, Medtronic, etc.)
- Advanced feature extraction using signal processing techniques
- Multi-modal insights combining physiological data with contextual information
- Personalized treatment plan generation based on goals and physiological responses
- Trend analysis and anomaly detection in longitudinal data
- Mobile app integration for real-time insights

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DiaTrend dataset creators for providing valuable real-world diabetes data
- OpenAI for providing the API used for generating insights
- The wearable device community for advancing health monitoring technology

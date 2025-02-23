# AI Refusal Crawler

This is an experimental implementation from the idea of the ARBOR Project: https://github.com/ArborProject/arborproject.github.io/discussions/5


A tool for systematically exploring and mapping the semantic graph structure of topics that trigger refusal behaviors in large language models (LLMs) using thought token forcing techniques.

## Overview

The AI Refusal Crawler is designed to:
- Discover and map topics that trigger refusal responses in LLMs
- Build semantic relationships between sensitive topics
- Track the evolution of refusal patterns
- Generate visualizations of the refusal topic space

## Features

- 🤖 Thought token forcing for topic discovery
- 🔍 Multi-model semantic embedding with GPU support
  - Primary: multilingual-e5-large-instruct
  - Secondary: all-MiniLM-L6-v2
  - Tertiary: LaBSE
- 📊 Graph-based topic relationship tracking
- 💾 Progress saving and resumption
- 📝 Detailed logging and analytics
- ⚡ Asynchronous operation for better performance
- 🎯 Ensemble-based similarity matching
- 🖥️ Model hosting using NDIF Cluster

## Requirements

- Python 3.8+
- OpenAI API key
- CUDA-capable GPU (optional, but recommended)
- Required Python packages (see `requirements.txt`)
- At least 8GB GPU memory for all models (4GB minimum for reduced setup)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-refusal-crawler.git
cd ai-refusal-crawler
```

2. Install main dependencies:
```bash
pip install -r requirements.txt
```

3. Install visualization dependencies:
```bash
pip install -r src/visualization/requirements.txt
```

4. Copy the example config and update with your settings:
```bash
cp config/config.yaml config/config.local.yaml
```

5. Update `config/config.local.yaml` with your:
   - NDIF host and API key
   - GPU settings (CUDA device selection)
   - Model configurations

## Usage

### Running the Crawler

1. Basic crawler usage:
```bash
python main.py --config config/config.yaml
```

2. Run with debug logging:
```bash
python main.py --config config/config.yaml --debug
```

### Running the Visualization Dashboard

1. Start the Streamlit dashboard:
```bash
streamlit run src/visualization/dashboard.py
```

The dashboard will be available at `http://localhost:8501` and provides:
- Interactive network visualization of the refusal graph
- Graph metrics and statistics
- Downloadable node data
- Timeline view of topic discovery

### Monitoring Progress

- Check `data/crawl_history/crawler.log` for real-time progress
- View discovered topics in `data/crawl_history/known_topics_*.txt`
- Examine graph snapshots in `data/graphs/refusal_graph_*.json`
- Use the visualization dashboard for interactive exploration

## Configuration

Key configuration options in `config.yaml`:

## Project Structure

```
refusal-crawler/
├── config/
│   └── config.yaml       # Configuration settings
├── src/
│   ├── crawler/         # Core crawler components
│   ├── analysis/        # Analysis components
│   ├── utils/           # Utility functions
│   └── visualization/   # Visualization components
├── data/
│   ├── crawl_history/   # Logs and history
│   └── graphs/          # Graph snapshots
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── README.md            # This file
├── docs/                # Documentation and project plan
└── playground/          # Folder for dummy experiment code

```


## License

MIT License

## Disclaimer

This tool is for research purposes only. Please use responsibly and in accordance with API providers' terms of service. 
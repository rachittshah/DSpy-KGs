# dspy-neo4j-knowledge-graph
LLM-driven automated knowledge graph construction from text using DSPy and Neo4j.

![Knowledge Graph Example](./example.png)

## Project Structure
```sh
dspy-neo4j-knowledge-graph/
├── README.md
├── requirements.txt
├── app.py
```

## Description
Model entities and relationships and build a Knowledge Graph using DSPy, Neo4j, and OpenAI's GPT-4. When given a paragraph or block of text, the app uses the DSPy library and OpenAI's GPT-4 to extract entities and relationships and generate a Cypher statement which is run in Neo4j to create the Knowledge Graph.

### Optimized Schema Context
The current graph schema is passed to the model as a list of nodes, relationships and properties in the context of the prompt. This allows the model to use elements from the existing schema and make connections between existing entities and relationships.

## Quick Start
1. Clone the repository.
2. Create a [Python virtual environment](#python-virtual-environment) and install the required packages.
3. Create a `.env` file and add the required [environment variables](#environment-variables).
4. [Run Neo4j using Docker](#usage).
5. Run `python3 run.py` and paste your text in the prompt.
6. Navigate to `http://localhost:7474/browser/` to view the Knowledge Graph in Neo4j Browser.

## Installation

### Prerequisites
* Python 3.12
* OpenAI API Key
* Docker

### Environment Variables
Before you begin, make sure to create a `.env` file and add your OpenAI API key.
```sh
NEO4J_URI=bolt://localhost:7687
OPENAI_API_KEY=<your-api-key>
```
### Next steps:

- Allow generation of KGs from documents. 
- Work on HybridRAG with KGs and RAG
- Evolve this into an KG eval toolkit. 

### Python Virtual Environment
Create a Python virtual environment and install the required packages.
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Run Neo4j using Docker.
```sh
docker run \
    --name dspy-kg \
    --publish=7474:7474 \
    --publish=7687:7687 \
    --env "NEO4J_AUTH=none" \
    neo4j:5.15
```

## Clean Up
Stop and remove the Neo4j container.
```sh
docker stop dspy-kg
docker rm dspy-kg
```

Deactivate the Python virtual environment.
```sh
deactivate
rm -rf .venv
```

## References
- [DSPy docs](https://dspy-docs.vercel.app/docs/intro)
- [Neo4j docs](https://neo4j.com/docs/)

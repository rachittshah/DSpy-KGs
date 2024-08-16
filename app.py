import os
import streamlit as st
from dotenv import find_dotenv, load_dotenv
import dspy
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import uuid

load_dotenv(find_dotenv())

class Neo4j:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

    def fmt_schema(self):
        query = """
        CALL db.schema.visualization()
        """
        schema = self.query(query)
        nodes = [node['name'] for node in schema[0]['nodes']]
        relationships = [f"{rel[0]['name']}-{rel[1]}->{rel[2]['name']}" for rel in schema[0]['relationships']]
        return f"Nodes: {', '.join(nodes)}\nRelationships: {', '.join(relationships)}"

    def get_graph_data(self):
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        return self.query(query)

neo4j = Neo4j(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

lm = dspy.OpenAI(
    model="gpt-4",
    max_tokens=1024,
)
dspy.configure(lm=lm)

class CypherFromText(dspy.Signature):
    """Instructions:
    Create a Cypher MERGE statement to model all entities and relationships found in the text following these guidelines:
    - Refer to the provided schema and use existing or similar nodes, properties or relationships before creating new ones.
    - Use generic categories for node and relationship labels.
    - Ensure the Cypher statement begins with 'MERGE' and does not include any explanatory text."""

    text = dspy.InputField(desc="Text to model using nodes, properties and relationships.")
    neo4j_schema = dspy.InputField(desc="Current graph schema in Neo4j as a list of NODES and RELATIONSHIPS.")
    statement = dspy.OutputField(desc="Cypher statement to merge nodes and relationships found in the text.")

generate_cypher = dspy.ChainOfThought(CypherFromText)

def process_text(text):
    try:
        cypher = generate_cypher(text=text.replace("\n", " "), neo4j_schema=neo4j.fmt_schema())
        statement = cypher.statement.replace('```', '').strip()
        if not statement.upper().startswith('MERGE'):
            raise ValueError("Generated Cypher statement does not start with 'MERGE'")
        result = neo4j.query(statement)
        return f"Success! Cypher statement executed", result
    except Exception as e:
        return f"Error: {str(e)}", None

def get_node_id(node):
    # Try to get a unique identifier for the node, fallback to a generated UUID
    return node.get('name') or node.get('id') or str(uuid.uuid4())

def get_node_label(node):
    # Try to get a meaningful label for the node, fallback to its type or 'Unknown'
    return node.get('name') or node.get('type') or 'Unknown'

def create_graph_visualization():
    graph_data = neo4j.get_graph_data()
    nt = Network(notebook=True, width="100%", height="600px", bgcolor="#222222", font_color="white")
    
    for item in graph_data:
        source = item['n']
        target = item['m']
        relationship = item['r']
        
        # Add source node
        source_id = get_node_id(source)
        nt.add_node(source_id, label=get_node_label(source), title=str(source), color="#FFA500")
        
        # Add target node if it exists
        if target:
            target_id = get_node_id(target)
            nt.add_node(target_id, label=get_node_label(target), title=str(target), color="#00BFFF")
            
            # Add edge if relationship exists
            if relationship:
                nt.add_edge(source_id, target_id, title=type(relationship).__name__)

    nt.save_graph("graph.html")
    
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    return html

def main():
    st.set_page_config(layout="wide", page_title="Text to Knowledge Graph")
    
    st.title("Text to Knowledge Graph")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        text = st.text_area("Enter a paragraph of text:", height=150)
        if st.button("Process Text"):
            if text:
                message, result = process_text(text)
                st.write(message)
                if result:
                    st.json(result)
            else:
                st.write("Please enter some text to process.")
        
        st.subheader("Current Schema")
        st.text(neo4j.fmt_schema())
    
    with col2:
        st.subheader("Knowledge Graph Visualization")
        try:
            html = create_graph_visualization()
            components.html(html, height=600)
        except Exception as e:
            st.error(f"Error creating graph visualization: {str(e)}")

if __name__ == "__main__":
    main()

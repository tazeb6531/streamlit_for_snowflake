
import pandas as pd
import webbrowser
import urllib.parse

df = pd.read_csv('first_app/data/employees.csv')

# Initialize the string to store the edges
edges = ""
nodes = set()  # To store unique nodes and avoid duplicates
for _, row in df.iterrows():
    if not pd.isna(row.iloc[1]):
        manager = row.iloc[0]
        employee = row.iloc[1]
        nodes.add(manager)
        nodes.add(employee)
        edges += f'"{manager}" -> "{employee}"\n'

# Initialize nodes string with special attributes
nodes_str = ""
for node in nodes:
    if node == "Hermann Baer":
        nodes_str += f'n1 [label = "{node}" shape = rect color=red];\n'
    else:
        nodes_str += f'"{node}" [label = "{node}"];\n'

# Create the final DOT format string
d = f'digraph d {{\n    graph [rankdir = BT];\n    node [];\n    edge [arrowhead = none];\n    {nodes_str}    {edges}}}'

# Encode the URL
url = f'http://magjac.com/graphviz-visual-editor/?dot={urllib.parse.quote(d)}'
webbrowser.open(url)






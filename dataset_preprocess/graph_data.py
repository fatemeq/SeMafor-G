import networkx as nx

# Replace 'docs.csv' with the actual path to your file
filename = '../dataset/docs.csv'

# Create an empty graph
G = nx.Graph()

# Output CSV filename (change as needed)
output_filename = '../dataset/graph_data.csv'

# Threshold for score to create an edge
score_threshold = 0.5

# Open CSV file for reading and output file for writing
with open(filename, 'r') as f_in, open(output_filename, 'w') as f_out:
  # Skip the header row (assuming your file has a header)
  next(f_in)
  f_out.write("Source,Target,Weight\n")  # Write header for output CSV

  for line in f_in:
    # Split the line by delimiter (usually comma)
    doc1_id, doc2_id, score = line.strip().split(',')

    # Convert score to float
    score = float(score)

    # Add nodes (always add nodes, even without edges)
    G.add_node(doc1_id)
    G.add_node(doc2_id)

    # Add edge only if score is above the threshold
    if score > score_threshold:
        G.add_edge(doc1_id, doc2_id, weight=score)
        f_out.write(f"{doc1_id},{doc2_id},{score}\n")

# Print the graph structure (optional)
# print(G.nodes())
# print(G.edges())



# from model.model import SemanticSimilarityModel

# model = SemanticSimilarityModel()

# # Input texts
# document1 = "But other sources close to the sale said Vivendi was keeping the door open to further bids and hoped to see bidders interested in individual assets team up."
# document2 = "But other sources close to the sale said Vivendi was keeping the door open for further bids in the next day or two."

# print('score: ' + str(model.similarity_score(document1, document2)))


import pandas as pd
import random
from model.model import SemanticSimilarityModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# document1 = """
# Architecture is the art and technique of designing and building, as distinguished from the skills associated with construction.[3] It is both the process and the product of sketching, conceiving,[4] planning, designing, and constructing buildings or other structures.[5] The term comes from Latin architectura; from Ancient Greek ἀρχιτέκτων (arkhitéktōn) 'architect'; from ἀρχι- (arkhi-) 'chief', and τέκτων (téktōn) 'creator'. Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilisations are often identified with their surviving architectural achievements.[6]

# The practice, which began in the prehistoric era, has been used as a way of expressing culture by civilizations on all seven continents.[7] For this reason, architecture is considered to be a form of art. Texts on architecture have been written since ancient times. The earliest surviving text on architectural theories is the 1st century AD treatise De architectura by the Roman architect Vitruvius, according to whom a good building embodies firmitas, utilitas, and venustas (durability, utility, and beauty). Centuries later, Leon Battista Alberti developed his ideas further, seeing beauty as an objective quality of buildings to be found in their proportions. In the 19th century, Louis Sullivan declared that "form follows function". "Function" began to replace the classical "utility" and was understood to include not only practical but also aesthetic, psychological and cultural dimensions. The idea of sustainable architecture was introduced in the late 20th century.

# Architecture began as rural, oral vernacular architecture that developed from trial and error to successful replication. Ancient urban architecture was preoccupied with building religious structures and buildings symbolizing the political power of rulers until Greek and Roman architecture shifted focus to civic virtues. Indian and Chinese architecture influenced forms all over Asia and Buddhist architecture in particular took diverse local flavors. During the Middle Ages, pan-European styles of Romanesque and Gothic cathedrals and abbeys emerged while the Renaissance favored Classical forms implemented by architects known by name. Later, the roles of architects and engineers became separated.

# Modern architecture began after World War I as an avant-garde movement that sought to develop a completely new style appropriate for a new post-war social and economic order focused on meeting the needs of the middle and working classes. Emphasis was put on modern techniques, materials, and simplified geometric forms, paving the way for high-rise superstructures. Many architects became disillusioned with modernism which they perceived as ahistorical and anti-aesthetic, and postmodern and contemporary architecture developed. Over the years, the field of architectural construction has branched out to include everything from ship design to interior decorating.
# """

# document2 = """
# Software architecture is the set of structures needed to reason about a software system and the discipline of creating such structures and systems. Each structure comprises software elements, relations among them, and properties of both elements and relations.[1][2]

# The architecture of a software system is a metaphor, analogous to the architecture of a building.[3] It functions as the blueprints for the system and the development project, which project management can later use to extrapolate the tasks necessary to be executed by the teams and people involved.

# Software architecture design is commonly juxtaposed with software application design. Whilst application design focuses on the design of the processes and data supporting the required functionality (the services offered by the system), software architecture design focuses on designing the infrastructure within which application functionality can be realized and executed such that the functionality is provided in a way which meets the system's non-functional requirements.

# Software architecture is about making fundamental structural choices that are costly to change once implemented. Software architecture choices include specific structural options from possibilities in the design of the software.

# For example, the systems that controlled the Space Shuttle launch vehicle had the requirement of being very fast and very reliable. Therefore, an appropriate real-time computing language would need to be chosen. Additionally, to satisfy the need for reliability the choice could be made to have multiple redundant and independently produced copies of the program, and to run these copies on independent hardware while cross-checking results.

# Documenting software architecture facilitates communication between stakeholders, captures early decisions about the high-level design, and allows reuse of design components between projects.
# """

# model = SemanticSimilarityModel()
# similarity_score = model.similarity_score(document1, document2)
# print(f'Similarity Score: {similarity_score}')

# Assuming your CSV files are named 'similar.csv' and 'dissimilar.csv'
similar_csv_path = "./dataset/similar_docs.csv"
dissimilar_csv_path = "./dataset/non_similar_docs.csv"

# Read CSV files into DataFrames
similar_df = pd.read_csv(similar_csv_path)
dissimilar_df = pd.read_csv(dissimilar_csv_path)

# Select a random sample of 200 rows from each DataFrame
similar_sample = similar_df.sample(n=200, random_state=42)
dissimilar_sample = dissimilar_df.sample(n=200, random_state=42)

# Concatenate the two samples into a new DataFrame
combined_df = pd.concat([similar_sample, dissimilar_sample], ignore_index=True)

# Shuffle the rows in the combined DataFrame
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined DataFrame to a new CSV file
combined_csv_path = "./dataset/combined_dataset.csv"
combined_df.to_csv(combined_csv_path, index=False)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Document1', 'Document2', 'ActualScore', 'PredictedScore'])

# Assuming your model class is named SemanticSimilarityModel
model = SemanticSimilarityModel()

# Load the combined dataset CSV file
combined_df = pd.read_csv(combined_csv_path)

# Iterate through each row in the combined DataFrame and calculate similarity scores
for index, row in combined_df.iterrows():
    document1_id = row.iloc[0]
    document2_id = row.iloc[1]
    
    document1 = str(document1_id)
    document2 = str(document2_id)

    # Load the actual score from the original CSV file
    actual_score = row['score']

    # Calculate the predicted similarity score
    predicted_score = model.similarity_score(document1, document2)
    
    # Store the results in the results DataFrame
    results_df = results_df.append({'Document1': document1_id, 'Document2': document2_id, 'ActualScore': actual_score, 'PredictedScore': predicted_score}, ignore_index=True)

    print(f'Similarity Score for pair {index + 1} ({document1_id}, {document2_id}): Actual: {actual_score}, Predicted: {predicted_score}')

# Save the results DataFrame to a new CSV file
results_csv_path = "./dataset/results.csv"
results_df.to_csv(results_csv_path, index=False)


# Assuming df is your DataFrame containing the results
actual_scores = results_df['ActualScore'].values
predicted_scores = results_df['PredictedScore'].values

# Calculate metrics
mse = mean_squared_error(actual_scores, predicted_scores)
mae = mean_absolute_error(actual_scores, predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(actual_scores, predicted_scores)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

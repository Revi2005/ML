import csv
import io

# CSV data stored inside the program
csv_data = """Sky,AirTemp,Humidity,Wind,Water,Forecast,PlaySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
"""

# Read CSV data
data = list(csv.reader(io.StringIO(csv_data)))

# Remove header
concepts = [row[:-1] for row in data[1:]]
targets = [row[-1] for row in data[1:]]

print("Training Data:\n")
for row in data:
    print(row)

# Initialize Specific Hypothesis
for i in range(len(targets)):
    if targets[i] == "Yes":
        S = concepts[i].copy()
        break

# Initialize General Hypothesis
G = [['?' for _ in range(len(S))]]

print("\nInitial Specific Hypothesis (S):", S)
print("Initial General Hypothesis (G):", G)

# Candidate Elimination Algorithm
for i, concept in enumerate(concepts):

    if targets[i] == "Yes":
        # Generalize S
        for j in range(len(S)):
            if S[j] != concept[j]:
                S[j] = '?'

        # Remove inconsistent hypotheses from G
        G = [g for g in G if all(g[k] == '?' or g[k] == S[k] for k in range(len(S)))]

    else:
        # Specialize G
        new_G = []
        for g in G:
            for j in range(len(S)):
                if g[j] == '?':
                    if S[j] != '?':
                        new_h = g.copy()
                        new_h[j] = S[j]
                        new_G.append(new_h)
        G = new_G

    print("\nAfter Training Example", i + 1)
    print("Specific Hypothesis (S):", S)
    print("General Hypothesis (G):", G)

print("\nFinal Specific Hypothesis:")
print(S)

print("\nFinal General Hypothesis:")
for g in G:
    print(g)
import pandas as pd

# ---- Step 1: Load dataset ----
# Example dataset: EnjoySport

data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=[
    'Sky', 'AirTemp', 'Humidity', 'Wind',
    'Water', 'Forecast', 'EnjoySport'
])

print("Dataset:\n", data)

# Attributes and target
attributes = data.columns[:-1]
target = data.columns[-1]


# ---- Step 2: Find-S Algorithm ----
def find_s(df):
    # Start with most specific hypothesis
    hypothesis = ['0'] * (len(df.columns) - 1)

    for i, row in df.iterrows():
        if row[target] == "Yes":  # positive example

            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = row.iloc[j]

                elif hypothesis[j] != row.iloc[j]:
                    hypothesis[j] = '?'

    return hypothesis


final_hypothesis = find_s(data)

print("\nMost specific hypothesis (Find-S):", final_hypothesis)


# ---- Step 3: Candidate Elimination Algorithm ----
def candidate_elimination(df):

    # Initialize S (specific) and G (general)
    S = [['0'] * (len(df.columns) - 1)]
    G = [['?'] * (len(df.columns) - 1)]

    for i, row in df.iterrows():

        if row[target] == "Yes":  # positive

            # Remove from G any inconsistent hypothesis
            G = [
                g for g in G
                if all(
                    g[j] == '?' or g[j] == row.iloc[j]
                    for j in range(len(attributes))
                )
            ]

            # Update S
            for j in range(len(attributes)):
                if S[0][j] == '0':
                    S[0][j] = row.iloc[j]

                elif S[0][j] != row.iloc[j]:
                    S[0][j] = '?'

        else:  # negative

            # Remove from S any inconsistent hypothesis
            S = [
                s for s in S
                if not all(
                    s[j] == '?' or s[j] == row.iloc[j]
                    for j in range(len(attributes))
                )
            ]

            # Specialize G
            new_G = []

            for g in G:
                for j in range(len(attributes)):

                    if g[j] == '?':
                        for val in df[attributes[j]].unique():

                            if val != row.iloc[j]:
                                new_hypo = g.copy()
                                new_hypo[j] = val

                                if any(
                                    all(
                                        s[k] == '?'
                                        or s[k] == new_hypo[k]
                                        or s[k] == '0'
                                        for k in range(len(attributes))
                                    )
                                    for s in S
                                ):
                                    new_G.append(new_hypo)

            G = new_G

    return S, G


S_final, G_final = candidate_elimination(data)

print("\nCandidate Elimination results:")
print("S (Most specific boundary):", S_final)
print("G (Most general boundary):", G_final)
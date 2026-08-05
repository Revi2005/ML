# FIND-S Algorithm

# Training dataset
# Attributes: Sky, AirTemp, Humidity, Wind, Water, Forecast, PlaySport

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Number of attributes (excluding class label)
num_attributes = len(data[0]) - 1

# Initialize hypothesis with the first positive example
hypothesis = ['0'] * num_attributes

print("Training Examples:")
for row in data:
    print(row)

print("\nApplying FIND-S Algorithm...\n")

first_positive = True

for example in data:
    if example[-1] == "Yes":

        if first_positive:
            hypothesis = example[:-1]
            first_positive = False
        else:
            for i in range(num_attributes):
                if hypothesis[i] != example[i]:
                    hypothesis[i] = '?'

        print("Current Hypothesis:", hypothesis)

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)
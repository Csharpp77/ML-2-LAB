import random


# Step 1: Generate a random number between 0 and 1
def uniform_random():
    return random.uniform(0, 1)


# Step 2: Define the function f(x) = x^2
def f(x):
    return x ** 2


# Step 3: Generate 10 random samples
N = 10
samples = [uniform_random() for _ in range(N)]


# Step 4: Evaluate the function
f_values = [f(x) for x in samples]

print("Random Samples (x):")
print(samples)

print("\nFunction values f(x) = x^2:")
print(f_values)


# Step 5: Calculate the Monte Carlo estimate
I_est = sum(f_values) / N

print("\nMonte Carlo Estimate of integral =", I_est)


# Step 6: Compare with the true value
I_true = 1 / 3

print("True Value =", I_true)
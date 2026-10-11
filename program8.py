from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# Step 1: Define the Bayesian Network structure
model = DiscreteBayesianNetwork([
    ("T", "S"),
    ("S", "F")
])


# Step 2: Define CPDs

cpd_t = TabularCPD(
    variable="T",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"T": ["Normal", "High"]}
)

cpd_s = TabularCPD(
    variable="S",
    variable_card=2,
    values=[
        [0.9, 0.4],
        [0.1, 0.6]
    ],
    evidence=["T"],
    evidence_card=[2],
    state_names={
        "S": ["No", "Yes"],
        "T": ["Normal", "High"]
    }
)

cpd_f = TabularCPD(
    variable="F",
    variable_card=2,
    values=[
        [0.8, 0.2],
        [0.2, 0.8]
    ],
    evidence=["S"],
    evidence_card=[2],
    state_names={
        "F": ["No", "Yes"],
        "S": ["No", "Yes"]
    }
)


# Step 3: Add CPDs to the model
model.add_cpds(cpd_t, cpd_s, cpd_f)


# Step 4: Check if the model is valid
print("Is model valid?", model.check_model())


# Step 5: Create inference object
inference = VariableElimination(model)


# Step 6: Perform inference queries

print("\nP(Fever | Temperature = High):")
print(
    inference.query(
        variables=["F"],
        evidence={"T": "High"}
    )
)

print("\nP(Fever | Sore Throat = Yes):")
print(
    inference.query(
        variables=["F"],
        evidence={"S": "Yes"}
    )
)

print("\nJoint probability of Temperature and Fever:")
print(
    inference.query(
        variables=["T", "F"]
    )
)
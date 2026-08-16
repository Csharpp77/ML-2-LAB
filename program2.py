import pandas as pd
from sklearn.datasets import load_iris
import wittgenstein as lw


# ---- Step 1: Load dataset (Iris for demo) ----
iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

df = pd.concat([X, y.rename("target")], axis=1)

print("Dataset head:\n", df.head())


# ---- Step 2: RIPPER Algorithm (via wittgenstein) ----
model = lw.RIPPER()

# Specify the positive class: class 0 (setosa)
model.fit(X, y, pos_class=0)

print("\n=== RIPPER Rules ===")
print(model.ruleset_)


# ---- Step 3: Simplified FOIL Implementation ----
# FOIL tries to build rules like "IF ... THEN class"
# We demonstrate binary classification: setosa vs not-setosa

df_bin = df.copy()

# 1 = setosa, 0 = other classes
df_bin["target"] = (df_bin["target"] == 0).astype(int)


# Candidate literals
attributes = list(X.columns)


def foil_gain(pos_before, neg_before, pos_after, neg_after):
    # FOIL information gain formula
    if pos_after == 0:
        return -1e9

    return pos_after * (
        (pos_after / (pos_after + neg_after))
        - (pos_before / (pos_before + neg_before))
    )


def foil(df, target_col="target"):

    rules = []

    pos_total = df[target_col].sum()
    neg_total = len(df) - pos_total

    while pos_total > 0:

        rule = []

        pos_rem = pos_total
        neg_rem = neg_total

        covered = df.copy()

        while neg_rem > 0:

            best_gain = -1e9
            best_attr = None
            best_val = None
            best_subset = None

            for attr in attributes:

                for val in df[attr].unique():

                    subset = covered[covered[attr] == val]

                    pos_after = subset[target_col].sum()
                    neg_after = len(subset) - pos_after

                    gain = foil_gain(
                        pos_rem,
                        neg_rem,
                        pos_after,
                        neg_after
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_attr = attr
                        best_val = val
                        best_subset = subset

            if best_attr is None:
                break

            rule.append((best_attr, best_val))

            covered = best_subset

            pos_rem = covered[target_col].sum()
            neg_rem = len(covered) - covered[target_col].sum()

        rules.append((rule, 1))

        # Remove examples covered by this rule
        df = df.drop(covered.index)

        pos_total = df[target_col].sum()
        neg_total = len(df) - pos_total

    return rules


# ---- Step 4: Apply FOIL ----
rules = foil(df_bin)

print("\n=== FOIL Learned Rules (Setosa vs Not) ===")

for conds, pred in rules:

    cond_str = " AND ".join(
        [f"{a}={v}" for a, v in conds]
    )

    print(f"IF {cond_str} THEN class={pred}")
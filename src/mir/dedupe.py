import pandas as pd
from thefuzz import fuzz
from typing import cast, Dict, List

# Read the dataset
df = pd.read_csv("./../../data/output.csv")

# Normalize the feature and target columns
df["feature_clean"] = df["feature"].str.lower().str.strip()
df["target"] = df["target"].str.strip()

# Remove duplicates based on the cleaned feature and target columns
df = df.drop_duplicates(subset=["feature_clean", "target"]).reset_index(drop=True)


def fuzzy_deduplicate(features: List[str], threshold: int = 90) -> Dict[str, List[str]]:
    """
    Group a list of feature strings based on fuzzy matching.
    Returns a dictionary that maps a "canonical" value to a list of
    variants that match (where scorer >= threshold).
    """
    canonical: Dict[str, List[str]] = {}
    for feat in features:
        match_found = False
        for key in canonical.keys():
            ratio = fuzz.token_sort_ratio(feat, key)
            if ratio >= threshold:
                canonical[key].append(feat)
                match_found = True
                break
        if not match_found:
            canonical[feat] = [feat]
    return canonical


# We create a mapping per target so that we don't mix groups across categories.
canonical_mapping: Dict[str, str] = {}

for target_val, grp in df.groupby("target"):
    unique_features = list(grp["feature_clean"].unique())
    # Adjust threshold depending on the leniency you want
    dup_dict = fuzzy_deduplicate(unique_features, threshold=80)
    for canonical_value, variants in dup_dict.items():
        for variant in variants:
            canonical_mapping[variant] = canonical_value

# Create a new column with the canonical (fuzzy deduplicated) version
df["canonical_feature"] = df["feature_clean"].map(lambda x: canonical_mapping.get(x, x))

# If you want to simply remove fuzzy duplicates (i.e. keep only one row per group)
# we drop duplicates based on the canonical feature and target.
df_unique = df.drop_duplicates(subset=["canonical_feature", "target"]).reset_index(
    drop=True
)

# (Optionally) update the "feature" column to use the canonical version
df_unique["feature"] = df_unique["canonical_feature"]

# Create the final DataFrame with the original columns.
# Using .loc and .copy() ensures that Pyright treats this as a DataFrame.
df_final: pd.DataFrame = df_unique.loc[:, ["feature", "target"]].copy()

# Save the cleaned dataset
output_file = "./../../data/cleaned_dataset.csv"
df_final.to_csv(output_file, index=False)

print("Cleaning of the dataset is complete. Duplicates have been removed.")
print(f"The cleaned dataset has been saved as '{output_file}'.")


def print_feature_to_target_ratio(df: pd.DataFrame) -> None:
    """
    For each target, print:
      - The count of unique features (after fuzzy deduplication).
      - The ratio of that target’s feature count to the total unique feature count.
    """
    # Group by target and count unique features, then cast the result as a Series.
    target_feature_counts: pd.Series = cast(
        pd.Series, df.groupby("target")["feature"].nunique()
    )
    total_unique_features: int = int(df["feature"].nunique())

    print("\nFeature to Target Ratio:")
    for target, count in target_feature_counts.items():
        ratio: float = (
            count / total_unique_features if total_unique_features != 0 else 0
        )
        print(f"Target: {target:<20} Count: {count:<4}  Ratio: {ratio:.2f}")


# Calling the function to print the counts in the console
print_feature_to_target_ratio(df_final)

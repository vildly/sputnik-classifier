import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your ragas results CSV file
csv_path = "output/ragas_20250318_211315/metrics_summary.csv"

# Load CSV into pandas DataFrame
df = pd.read_csv(csv_path)
print("Data preview:")
print(df.head())

# Identify columns that are numeric metrics (assuming 'response' and 'reference' are non-numeric).
metric_cols = [col for col in df.columns if col not in ['response', 'reference']]
print("Metric columns:", metric_cols)


# --------------------------------------------------
# 1. Violin Plot for Distribution of All Ragas Evaluation Metrics
# --------------------------------------------------
df_melted = df.melt(id_vars=['response', 'reference'], value_vars=metric_cols, 
                    var_name='metric', value_name='score')

plt.figure(figsize=(12, 6))
sns.violinplot(x='metric', y='score', data=df_melted, inner='quartile', palette='Pastel1')
plt.title("Distribution of All Ragas Evaluation Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Grouped Metrics by Reference Category (Bar Plot)
# --------------------------------------------------
grouped_metrics = df.groupby('reference')[metric_cols].mean().reset_index()

# Plot grouped bar chart for average scores
grouped_metrics.set_index('reference').plot(kind='bar', figsize=(12, 8), colormap="tab10")
plt.title("Mean Scores per Metric Grouped by Reference Category")
plt.ylabel("Mean Score")
plt.xlabel("Reference Category")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.legend(title="Ragas Metrics")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Violin Plot Grouped by Reference Category
# --------------------------------------------------
sns.violinplot(x='reference', y='semantic_similarity', data=df,
               inner='quartile', palette='Set2', bw=0.2)
plt.title("Distribution of Semantic Similarity by Reference Category")
plt.xlabel("Reference Category")
plt.ylabel("Semantic Similarity")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Enhanced Confusion Matrix (Omitting Zeros)
# --------------------------------------------------
# Create a confusion matrix with counts
confusion_matrix = pd.crosstab(df['reference'], df['response'])

# Mask out zero values
mask = confusion_matrix == 0

# Plot the enhanced heatmap with omitted zeroes
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_matrix,
    annot=confusion_matrix,  # Annotate non-zero values
    fmt='d',                # Integer formatting for counts
    cmap='Blues',           # Color map for the heatmap
    mask=mask,              # Mask cells with 0
    cbar=False
)

# Add labels and title
plt.title("Enhanced Confusion Matrix (Non-Zero Values Only)")
plt.xlabel("Predicted Category")
plt.ylabel("True Category")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Overall Classification Rates (TPR & TNR)
# --------------------------------------------------
# Total number of samples
total_samples = len(df)

# Calculate True Positives (TP) and True Negatives (TN)
tp = sum(df['reference'] == df['response'])  # Correctly predicted categories
tn = total_samples - tp  # Everything else is a true negative

# Calculate TPR and TNR
tpr = tp / total_samples  # True Positive Rate
tnr = tn / total_samples  # True Negative Rate

# Create a confusion-matrix-like 2x2 table
matrix = np.array([[tpr, 1 - tpr], [1 - tnr, tnr]])

# Annotate the matrix for display (e.g., percentages)
annot = np.array([
    [f"TPR = {tpr:.2f}", f"FNR = {1-tpr:.2f}"],  # Row: True Positives/Negatives
    [f"FPR = {1-tnr:.2f}", f"TNR = {tnr:.2f}"]   # Row: False Negatives/Positives
])

# Plot the matrix
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=annot, fmt="", cmap="YlGnBu", cbar=False, xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.title("Overall Classification Rates (TPR & TNR)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

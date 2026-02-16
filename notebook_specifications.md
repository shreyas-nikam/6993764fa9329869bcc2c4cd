
# Individual Fairness Audit: Twin Applicant Credit Review

## Case Study Introduction

As a **CFA Charterholder and Risk Manager** at **CreditGuard Financial**, a leading institution committed to ethical AI practices, my primary responsibility is to ensure that our automated credit scoring models operate with the highest standards of fairness and compliance. Regulators, including those enforcing the Equal Credit Opportunity Act (ECOA), are increasingly scrutinizing AI models for individual-level discrimination, not just group disparities. My team recently deployed an AI-powered credit model, and while it has demonstrated strong group-level fairness metrics, I need to proactively audit it for individual-level unfairness – instances where similar individuals receive different outcomes.

This notebook documents my workflow to conduct a rigorous individual fairness audit using 'twin applicant' testing, SHAP explanations, and Lipschitz fairness measurements. My goal is to identify and quantify any potential individual discrimination, understand its drivers, and assess the model's overall sensitivity to non-financial features. This proactive approach helps **CreditGuard Financial** mitigate regulatory risk, enhance our ethical reputation, and improve model robustness.

---

### 1. Initial Setup: Installing Libraries and Loading Data

To begin our individual fairness audit, we need to set up our environment by installing the necessary Python libraries and loading the pre-trained credit scoring model along with the applicant dataset. This ensures we have all the tools and data ready for constructing twin applicants and evaluating model fairness.

```python
# Install required libraries
!pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn --quiet
!pip install ipython # For display in Twin Applicant Gallery

# Import required dependencies
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, HTML

# --- Pre-trained Model and Data Simulation ---
# In a real scenario, these would be loaded from persistence/databases.
# For this lab, we'll simulate a dataset and train a placeholder model.

# 1. Simulate a Credit Loan Applicant Dataset
np.random.seed(42)
n_samples = 5000

data = {
    'FICO_score': np.random.randint(600, 850, n_samples),
    'income': np.random.randint(40000, 150000, n_samples),
    'debt_to_income_ratio': np.random.uniform(0.1, 0.5, n_samples),
    'loan_amount': np.random.randint(5000, 50000, n_samples),
    'employment_length_years': np.random.randint(0, 20, n_samples),
    'revolving_utilization': np.random.uniform(0.1, 0.9, n_samples), # Proxy feature
    'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.3, 0.3, 0.4]),
    'ZIP_code': np.random.choice(['10001', '10025', '10456', '90210', '77001'], n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]), # Proxy feature
    'default': np.random.randint(0, 2, n_samples) # Target variable (0: no default, 1: default)
}
df = pd.DataFrame(data)

# Introduce some correlation with proxy features for demonstration
# E.g., higher default rates for certain ZIP codes or revolving utilization
df.loc[df['ZIP_code'] == '10456', 'default'] = np.random.choice([0, 1], sum(df['ZIP_code'] == '10456'), p=[0.4, 0.6])
df.loc[df['revolving_utilization'] > 0.7, 'default'] = np.random.choice([0, 1], sum(df['revolving_utilization'] > 0.7), p=[0.3, 0.7])

# Encode categorical features
le = LabelEncoder()
df['home_ownership_encoded'] = le.fit_transform(df['home_ownership'])
df['ZIP_code_encoded'] = le.fit_transform(df['ZIP_code']) # Encode ZIP code for model

# Define features and target
financial_features = ['FICO_score', 'income', 'debt_to_income_ratio', 'loan_amount', 'employment_length_years']
proxy_features = ['revolving_utilization', 'home_ownership_encoded', 'ZIP_code_encoded']
feature_cols = financial_features + proxy_features
target_col = 'default'

X = df[feature_cols]
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a placeholder XGBoost Classifier (our "Credit Scoring Model")
# This model is what we will be auditing.
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
model.fit(X_train, y_train)

# For SHAP, we need a predict function that takes a numpy array
model_predict_proba = lambda x: model.predict_proba(x)[:, 1]

print("Setup complete: Dataset simulated and XGBoost model trained.")
print(f"Financial Features: {financial_features}")
print(f"Proxy Features: {proxy_features}")
print(f"Total Features used by model: {feature_cols}")
```

---

### 2. Task 1: Constructing Twin Applicants for Fair Auditing

My first step as a Risk Manager is to create 'twin applicant' pairs. This critical technique allows me to isolate the impact of potentially biased or proxy features on credit decisions. I construct hypothetical applicants who are identical in all financially relevant aspects but differ only in a single, non-financial, potentially proxy-correlated feature (like `ZIP_code` or `revolving_utilization`). By comparing their model predictions, I can directly observe if the model is treating similar individuals dissimilarly, which is a key indicator of individual unfairness.

This process is akin to the "matched pair" testing regulators have used for decades, now scaled with AI. It helps CreditGuard Financial identify subtle biases that aggregate group fairness metrics might overlook.

```python
def construct_twin_pairs(X_data, financial_features, proxy_features, n_pairs=50, seed=42):
    """
    Constructs 'twin applicant' pairs. Each pair consists of an original applicant
    and a twin. The twin is identical to the original across all financial features
    but has its specified proxy features perturbed to "opposite" typical values.

    Args:
        X_data (pd.DataFrame): The dataset from which to sample applicants.
        financial_features (list): List of column names representing financially relevant features.
        proxy_features (list): List of column names representing potentially proxy-correlated features.
        n_pairs (int): The number of twin pairs to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of dictionaries, where each dictionary contains 'original' (pd.Series),
              'twin' (pd.Series), 'original_idx', and 'proxy_changed' for the pair.
    """
    np.random.seed(seed)
    pairs = []
    
    # Randomly sample unique indices for original applicants
    sample_indices = np.random.choice(len(X_data), min(n_pairs, len(X_data)), replace=False)

    for idx in sample_indices:
        original = X_data.iloc[idx].copy()
        twin = original.copy()

        # Perturb only the specified proxy features
        perturbed_proxies = {}
        for proxy_feat in proxy_features:
            original_proxy_value = original[proxy_feat]
            col_values = X_data[proxy_feat]
            
            # Determine the "opposite" typical value for the proxy feature
            # If original is in the "high" group (above median), set twin to "low" (below median)
            # If original is in the "low" group (below median), set twin to "high" (above median)
            
            # Handle cases where median might not be representative (e.g., highly skewed discrete features)
            # For simplicity, we'll use a percentile-based split or just the opposite side of the median.
            # Here, we directly take the median of the 'opposite' group.
            
            # Check if there are values in the "opposite" group before calculating median
            if original_proxy_value > col_values.median():
                # Original is in "high" group; set twin to a typical "low" value
                opposite_group_values = col_values[col_values <= col_values.median()]
                if not opposite_group_values.empty:
                    twin[proxy_feat] = opposite_group_values.median()
                else: # Fallback if no values in opposite group
                    twin[proxy_feat] = col_values.min()
            else:
                # Original is in "low" group; set twin to a typical "high" value
                opposite_group_values = col_values[col_values > col_values.median()]
                if not opposite_group_values.empty:
                    twin[proxy_feat] = opposite_group_values.median()
                else: # Fallback if no values in opposite group
                    twin[proxy_feat] = col_values.max()
            
            perturbed_proxies[proxy_feat] = (original_proxy_value, twin[proxy_feat])

        pairs.append({
            'original': original,
            'twin': twin,
            'original_idx': idx,
            'proxy_changed': list(perturbed_proxies.keys()) # Store which proxies were varied
        })
    
    print(f"Constructed {len(pairs)} twin applicant pairs.")
    print(f"Proxy features varied: {proxy_features}")
    print("All other features (financial_features) are IDENTICAL between twins.")
    return pairs

# Execute the function to construct twin pairs
n_audit_pairs = 100 # Number of twin pairs to create for the audit
twin_applicant_pairs = construct_twin_pairs(X_test, financial_features, proxy_features, n_pairs=n_audit_pairs)

# Display a sample twin pair to verify construction
print("\n--- Sample Twin Pair Verification ---")
sample_pair = twin_applicant_pairs[0]
print(f"Original Applicant (Index {sample_pair['original_idx']}):")
display(sample_pair['original'].to_frame().T)
print("Twin Applicant:")
display(sample_pair['twin'].to_frame().T)
print(f"Features intentionally changed for twin: {sample_pair['proxy_changed']}")
```

After constructing `n_audit_pairs` twin applicants, I can see that the `FICO_score`, `income`, and `debt_to_income_ratio` remain constant for both the original and twin applicants, while the `revolving_utilization`, `home_ownership_encoded`, and `ZIP_code_encoded` have been modified. This deliberate perturbation allows us to isolate the model's sensitivity to these specific proxy features. The generated pairs are now ready to be fed into our credit scoring model to detect any differential treatment.

---

### 3. Task 2: Detecting Individual Fairness Violations

Now that I have my twin applicant pairs, my next critical step is to feed them into our credit scoring model and analyze its predictions. As a Risk Manager, I'm specifically looking for 'decision flips' – where one twin is approved and the other denied – or 'material prediction deltas' – a significant difference in approval probability, even if both are approved or denied. This directly quantifies individual unfairness.

The **Individual Fairness Violation Rate** measures the proportion of twin pairs where the absolute difference in prediction probabilities exceeds a predefined materiality threshold $\epsilon$. The **Decision Flip Rate** is a more severe metric, indicating instances where the model's final binary decision (approve/deny) changes between identical twins.

The mathematical formulations for these metrics are:

**Individual Fairness Violation Rate ($VR_{individual}$):**
$$VR_{individual} = \frac{|\{(x, x') : |f(x) - f(x')| > \epsilon, x \sim_{financial} x'\}|}{|\text{all twin pairs}|}$$
Where:
-   $x$ and $x'$ are a twin pair of applicants.
-   $x \sim_{financial} x'$ means $x$ and $x'$ have identical financial features but different proxy features.
-   $f(x)$ and $f(x')$ are the model's predicted credit approval probabilities for $x$ and $x'$, respectively.
-   $\epsilon$ is the materiality threshold for prediction difference (e.g., 0.10).
-   $|\cdot|$ denotes the count of elements in a set.

**Decision Flip Rate ($FR$):**
$$FR = \frac{|\{(x, x') : \hat{y}(x) \neq \hat{y}(x')\}|}{|\text{all twin pairs}|}$$
Where:
-   $\hat{y}(x)$ and $\hat{y}(x')$ are the model's binarized (e.g., using a 0.5 threshold) credit approval decisions for $x$ and $x'$, respectively.
-   $\hat{y}(x) \neq \hat{y}(x')$ indicates a decision flip.

By quantifying these rates, I can provide concrete evidence of individual unfairness to our compliance team and, if necessary, to regulators.

```python
def detect_individual_violations(model_predict_proba, twin_pairs, feature_cols, decision_threshold=0.5, materiality_threshold=0.10):
    """
    Detects individual fairness violations (prediction deltas and decision flips)
    for each twin pair.

    Args:
        model_predict_proba (callable): A function that takes a DataFrame or NumPy array
                                        of features and returns prediction probabilities (for class 1).
        twin_pairs (list): List of dictionaries, each containing 'original' and 'twin' applicant data.
        feature_cols (list): List of features (column names) used by the model.
        decision_threshold (float): Probability threshold to convert scores to binary decisions.
        materiality_threshold (float): Threshold for prediction difference to be considered a 'violation'.

    Returns:
        pd.DataFrame: A DataFrame summarizing detected violations, including
                      original/twin predictions, delta, and flags for flips/violations.
    """
    violations = []

    for pair in twin_pairs:
        original_features_df = pair['original'][feature_cols].to_frame().T
        twin_features_df = pair['twin'][feature_cols].to_frame().T

        orig_prob = model_predict_proba(original_features_df)[0]
        twin_prob = model_predict_proba(twin_features_df)[0]

        delta = abs(orig_prob - twin_prob)

        orig_decision = 1 if orig_prob >= decision_threshold else 0
        twin_decision = 1 if twin_prob >= decision_threshold else 0
        
        decision_flipped = (orig_decision != twin_decision)

        violations.append({
            'original_idx': pair['original_idx'],
            'orig_prob': round(orig_prob, 4),
            'twin_prob': round(twin_prob, 4),
            'delta': round(delta, 4),
            'orig_decision': orig_decision,
            'twin_decision': twin_decision,
            'decision_flipped': decision_flipped,
            'is_violation': delta > materiality_threshold,
            'proxy_changed': ', '.join(pair['proxy_changed'])
        })

    viol_df = pd.DataFrame(violations)
    
    n_violations = viol_df['is_violation'].sum()
    n_flipped = viol_df['decision_flipped'].sum()

    print("--- INDIVIDUAL FAIRNESS VIOLATION DETECTION ---")
    print("=" * 55)
    print(f"Twin pairs tested: {len(viol_df)}")
    print(f"Prediction delta > {materiality_threshold} (Violation Rate): {n_violations} ({n_violations / len(viol_df):.1%})")
    print(f"Decision flipped (Decision Flip Rate): {n_flipped} ({n_flipped / len(viol_df):.1%})")
    print(f"Mean prediction delta: {viol_df['delta'].mean():.4f}")
    print(f"Max prediction delta: {viol_df['delta'].max():.4f}")
    
    print("\nWORST VIOLATIONS (Top 5 by delta):")
    worst_violations = viol_df.nlargest(5, 'delta')
    for _, row in worst_violations.iterrows():
        flip_flag = "<<< DECISION FLIPPED >>>" if row['decision_flipped'] else ""
        print(f"Pair {row['original_idx']}: P(orig) {row['orig_prob']:.3f} -> P(twin) {row['twin_prob']:.3f} (delta={row['delta']:.3f}) {flip_flag} Proxy features changed: {row['proxy_changed']}")
        
    return viol_df

# Execute violation detection
violations_df = detect_individual_violations(model_predict_proba, twin_applicant_pairs, feature_cols, materiality_threshold=0.10)
```

**Output Explanation and Real-World Impact:**

The output provides concrete metrics for individual fairness. A **Violation Rate** of `X%` indicates that in `X%` of cases, similar applicants received significantly different credit approval probabilities. More critically, a **Decision Flip Rate** of `Y%` means `Y%` of twins, identical in all financial aspects, were given contradictory approve/deny decisions.

As a Risk Manager, these numbers are direct evidence of potential individual discrimination. A high Decision Flip Rate, especially, is a major red flag, potentially leading to **adverse action notices** and **regulatory scrutiny under ECOA**. If Applicant A (with ZIP 10001) is approved and Applicant B (identical except ZIP 10456) is denied, Applicant B has grounds to request an explanation. If the explanation "your ZIP code contributed to the decline" is problematic due to its correlation with protected attributes, CreditGuard Financial faces significant liability. This quantification highlights the need for further investigation and potential model refinement.

```python
# --- Visualization: Violation Rate Histogram ---
plt.figure(figsize=(10, 6))
sns.histplot(violations_df['delta'], bins=30, kde=True, color='skyblue')
plt.axvline(x=0.10, color='red', linestyle='--', label='Materiality Threshold ($\epsilon=0.10$)')
plt.axvspan(0.10, violations_df['delta'].max(), color='red', alpha=0.1, label='Individual Fairness Violations')
# Highlight decision flip region if distinct
decision_flip_deltas = violations_df[violations_df['decision_flipped']]['delta']
if not decision_flip_deltas.empty:
    min_flip_delta = decision_flip_deltas.min()
    plt.axvspan(min_flip_delta, violations_df['delta'].max(), color='purple', alpha=0.15, label='Decision Flip Region')

plt.title('Distribution of Prediction Deltas Across Twin Pairs', fontsize=14)
plt.xlabel('Absolute Prediction Probability Difference ($\Delta P$)', fontsize=12)
plt.ylabel('Number of Twin Pairs', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()
```

The histogram visually reinforces the numerical findings. I can clearly see the distribution of prediction differences, how many fall above our materiality threshold ($\epsilon=0.10$), and identify the region where decision flips occurred. This visualization is crucial for presenting the audit findings to non-technical stakeholders and regulatory bodies at CreditGuard Financial, making the concept of individual unfairness tangible.

---

### 4. Task 3: Decomposing Unfairness with SHAP

Identifying that a model produces unfair outcomes is one thing; understanding *why* is another. As a Risk Manager, after detecting individual fairness violations, my next step is to pinpoint the exact features driving these diverging predictions. For this, I will use **SHAP (SHapley Additive exPlanations)** values. SHAP helps decompose the prediction difference between an original applicant and their twin, attributing the impact to individual features.

This process transforms the abstract finding of "the model is unfair" into "this specific feature causes unfairness for this specific applicant." This level of detail is invaluable for CreditGuard Financial to:
1.  **Generate comprehensive Adverse Action Notices:** Explain to a denied applicant exactly why their profile led to that decision, even when a similar applicant was approved.
2.  **Inform model developers:** Provide actionable insights on which features are disproportionately influencing outcomes for similar individuals.
3.  **Bolster regulatory responses:** Demonstrate a deep understanding of model behavior and commitment to addressing bias at a granular level.

```python
def decompose_twin_difference(model, pair, feature_cols):
    """
    Uses SHAP to decompose the prediction difference for a twin pair,
    identifying which features drive the divergence.

    Args:
        model: The trained credit scoring model (e.g., XGBClassifier).
        pair (dict): A dictionary containing 'original' and 'twin' applicant data.
        feature_cols (list): List of features (column names) used by the model.

    Returns:
        tuple: (pd.DataFrame) Feature impact analysis for the pair,
               (pd.Series) The top contributing feature and its impact.
    """
    explainer = shap.TreeExplainer(model) # Use TreeExplainer for tree-based models

    original_features_df = pair['original'][feature_cols].to_frame().T
    twin_features_df = pair['twin'][feature_cols].to_frame().T

    # Calculate SHAP values for the original and twin applicant
    orig_shap_values = explainer.shap_values(original_features_df)[0] # For class 1 prediction
    twin_shap_values = explainer.shap_values(twin_features_df)[0] # For class 1 prediction

    shap_diff = twin_shap_values - orig_shap_values

    feature_impact = pd.DataFrame({
        'feature': feature_cols,
        'original_value': pair['original'][feature_cols].values,
        'twin_value': pair['twin'][feature_cols].values,
        'orig_shap': orig_shap_values,
        'twin_shap': twin_shap_values,
        'shap_diff': shap_diff,
        'abs_diff': np.abs(shap_diff)
    }).sort_values('abs_diff', ascending=False).reset_index(drop=True)

    # The top contributor is the "driver" of the unfair difference
    top_driver = feature_impact.iloc[0]

    return feature_impact, top_driver


def create_twin_applicant_gallery(model, worst_violations_df, twin_applicant_pairs, feature_cols, n_display=5):
    """
    Generates a visual gallery of the worst-case unfair twin pairs,
    displaying their features, predictions, and SHAP decompositions.
    """
    print(f"\n--- TWIN APPLICANT GALLERY (Top {n_display} Worst Violations) ---")
    print("=" * 60)

    gallery_html = ""

    # Sort by delta for consistency
    worst_violations = worst_violations_df.nlargest(n_display, 'delta')

    for i, (_, row) in enumerate(worst_violations.iterrows()):
        original_idx = row['original_idx']
        pair = next(p for p in twin_applicant_pairs if p['original_idx'] == original_idx)

        feature_impact, top_driver = decompose_twin_difference(model, pair, feature_cols)

        orig_df = pair['original'][feature_cols].to_frame().T
        twin_df = pair['twin'][feature_cols].to_frame().T

        # Prepare data for SHAP waterfall plots
        # We need the explainer.expected_value and shap_values for waterfall plot
        explainer = shap.TreeExplainer(model)
        
        # Get actual shap values for original and twin
        original_shap_values = explainer.shap_values(orig_df)[0]
        twin_shap_values = explainer.shap_values(twin_df)[0]
        
        expected_value = explainer.expected_value # Base value for the model's output

        # Create interactive SHAP waterfall plots (static image for notebook)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Original Applicant SHAP
        shap.waterfall_plot(shap.Explanation(values=original_shap_values, base_values=expected_value, data=orig_df.iloc[0], feature_names=feature_cols), max_display=10, show=False, ax=axes[0])
        axes[0].set_title(f'Original Applicant {original_idx} SHAP (P={row["orig_prob"]:.3f})')
        axes[0].set_xlabel('SHAP Value (impact on model output)')
        
        # Twin Applicant SHAP
        shap.waterfall_plot(shap.Explanation(values=twin_shap_values, base_values=expected_value, data=twin_df.iloc[0], feature_names=feature_cols), max_display=10, show=False, ax=axes[1])
        axes[1].set_title(f'Twin Applicant SHAP (P={row["twin_prob"]:.3f})')
        axes[1].set_xlabel('SHAP Value (impact on model output)')
        
        plt.tight_layout()
        
        # Save plot to buffer and embed as base64
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode()

        # Generate HTML table for features
        features_html = "<table><tr><th>Feature</th><th>Original Value</th><th>Twin Value</th></tr>"
        for col in feature_cols:
            orig_val = pair['original'][col]
            twin_val = pair['twin'][col]
            if col in financial_features:
                color = 'green' # Financial features should be identical
            elif col in proxy_features and orig_val != twin_val:
                color = 'red' # Proxy features intentionally different
            else:
                color = 'black' # Other/unchanged features
            features_html += f"<tr><td>{col}</td><td style='color:{color};'>{orig_val:.2f}</td><td style='color:{color};'>{twin_val:.2f}</td></tr>"
        features_html += "</table>"
        
        gallery_html += f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 20px;">
            <h3>Pair {i+1}: Original Index {original_idx} (Delta={row['delta']:.3f}) {("<<< DECISION FLIPPED >>>" if row['decision_flipped'] else "")}</h3>
            <p><strong>Original Prediction: {row['orig_prob']:.3f}</strong> | <strong>Twin Prediction: {row['twin_prob']:.3f}</strong></p>
            <p><strong>Primary Driver of Difference:</strong> {top_driver['feature']} (SHAP Diff: {top_driver['shap_diff']:.4f})</p>
            <div style="display: flex; justify-content: space-around;">
                <div style="width: 45%;">
                    <h4>Features Comparison</h4>
                    {features_html}
                </div>
                <div style="width: 50%;">
                    <h4>SHAP Waterfall Plots</h4>
                    <img src="data:image/png;base64,{img_str}" style="width: 100%;">
                </div>
            </div>
        </div>
        """
    display(HTML(gallery_html))

# Execute SHAP decomposition for worst violations and generate gallery
create_twin_applicant_gallery(model, violations_df, twin_applicant_pairs, feature_cols, n_display=5)

# --- Visualization: SHAP Waterfall Comparisons for the single WORST violation pair ---
print("\n--- SHAP Waterfall Comparisons for the SINGLE WORST VIOLATION PAIR ---")
print("=" * 70)

worst_pair_row = violations_df.nlargest(1, 'delta').iloc[0]
worst_original_idx = worst_pair_row['original_idx']
worst_pair_data = next(p for p in twin_applicant_pairs if p['original_idx'] == worst_original_idx)

explainer = shap.TreeExplainer(model)
worst_orig_df = worst_pair_data['original'][feature_cols].to_frame().T
worst_twin_df = worst_pair_data['twin'][feature_cols].to_frame().T

# Get actual shap values for original and twin
worst_original_shap_values = explainer.shap_values(worst_orig_df)[0]
worst_twin_shap_values = explainer.shap_values(worst_twin_df)[0]
expected_value = explainer.expected_value

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

shap.waterfall_plot(shap.Explanation(values=worst_original_shap_values, base_values=expected_value, data=worst_orig_df.iloc[0], feature_names=feature_cols), max_display=10, show=False, ax=axes[0])
axes[0].set_title(f'Worst Original Applicant {worst_original_idx} SHAP (P={worst_pair_row["orig_prob"]:.3f})', fontsize=14)
axes[0].set_xlabel('SHAP Value (impact on model output)', fontsize=12)

shap.waterfall_plot(shap.Explanation(values=worst_twin_shap_values, base_values=expected_value, data=worst_twin_df.iloc[0], feature_names=feature_cols), max_display=10, show=False, ax=axes[1])
axes[1].set_title(f'Worst Twin Applicant SHAP (P={worst_pair_row["twin_prob"]:.3f})', fontsize=14)
axes[1].set_xlabel('SHAP Value (impact on model output)', fontsize=12)

plt.tight_layout()
plt.suptitle(f'SHAP Waterfall Comparison for Worst Violation Pair {worst_original_idx}', fontsize=16, y=1.02)
plt.show()
```

The "Twin Applicant Gallery" provides a compelling, visual narrative for each worst-case violation. For each pair, I can observe the identical financial features (highlighted in green) and the differing proxy features (highlighted in red). Crucially, the side-by-side SHAP waterfall plots for the worst cases illuminate *how* specific feature contributions, especially from the varied proxy features, lead to the divergent predictions.

For example, if `ZIP_code_encoded` has a negative SHAP value for the original applicant (leading to approval) and a positive SHAP value for the twin (leading to denial), it unequivocally shows how that proxy feature drove the unfair outcome. This direct evidence is powerful for CreditGuard Financial to address regulatory concerns and improve model interpretability and fairness. The SHAP comparison for the single worst violation further emphasizes this by showing the exact shift in feature contributions.

---

### 5. Task 4: Measuring Model Sensitivity (Lipschitz Fairness)

Beyond specific twin pairs, I need to assess the model's overall robustness and sensitivity to minor perturbations in its input features. This is where **Lipschitz Fairness** comes in. The principle is that "similar individuals should receive similar predictions." A model is considered Lipschitz-fair if its prediction changes predictably (and not drastically) when input features change by a small amount.

The practical metric we use is the **Proxy/Financial Sensitivity Ratio (R)**. This ratio compares the model's average prediction sensitivity to changes in proxy features versus changes in legitimate financial features.

**Mathematical Formulation for Lipschitz Individual Fairness and Proxy/Financial Sensitivity Ratio:**

A model $f$ satisfies $(\epsilon, \delta)$-individual fairness if:
$$d_y(f(x_1), f(x_2)) \le L \cdot d_x(x_1, x_2)$$
Where:
-   $d_x(x_1, x_2)$ is a distance metric on input features (e.g., Euclidean distance).
-   $d_y(f(x_1), f(x_2))$ is the prediction distance (e.g., absolute difference in probabilities).
-   $L$ is the Lipschitz constant, representing the maximum rate of change in output for a unit change in input.

Financial Interpretation: For two applicants with similar credit profiles ($d_x$ small on financial features), the model's predictions should also be similar ($d_y$ small). If the prediction changes dramatically when only a proxy feature changes, the Lipschitz constant with respect to proxy features is high, indicating individual unfairness.

The **Proxy/Financial Sensitivity Ratio (R)** is our practical metric:
$$R = \frac{S_{proxy}}{S_{financial}}$$
Where:
-   $S_{proxy}$ is the average prediction sensitivity to small perturbations in proxy features.
-   $S_{financial}$ is the average prediction sensitivity to small perturbations in financial features.

**Interpretation:**
-   $R < 0.5$: Good. Model is much more sensitive to financial features than proxies.
-   $0.5 \le R < 1.0$: Warning. Proxy sensitivity is comparable.
-   $R \ge 1.0$: Fail. Model is more sensitive to proxies than to legitimate credit factors.

This analysis helps CreditGuard Financial understand if the model relies too heavily on proxy features even when minor variations occur, indicating a structural susceptibility to individual bias.

```python
def lipschitz_fairness(model_predict_proba, X_data, feature_cols, proxy_features, n_sample=500, perturbation_strength=0.1, seed=42):
    """
    Measures the model's sensitivity to small perturbations in each feature
    to assess Lipschitz fairness. Calculates the Proxy/Financial Sensitivity Ratio.

    Args:
        model_predict_proba (callable): A function that takes a DataFrame or NumPy array
                                        of features and returns prediction probabilities (for class 1).
        X_data (pd.DataFrame): The dataset to sample from for sensitivity analysis.
        feature_cols (list): List of all features used by the model.
        proxy_features (list): List of features considered as proxy/sensitive.
        n_sample (int): Number of samples to use for sensitivity calculation.
        perturbation_strength (float): Percentage of feature's standard deviation to perturb by.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (pd.DataFrame) DataFrame of feature sensitivities,
               (float) The calculated Lipschitz (Proxy/Financial) ratio.
    """
    np.random.seed(seed)
    
    # Sample data for sensitivity calculation
    sample_data = X_data.sample(min(n_sample, len(X_data)), random_state=seed)
    
    # Baseline predictions
    base_probs = model_predict_proba(sample_data[feature_cols])

    sensitivity_ratios = {}

    for feat in feature_cols:
        perturbed_data = sample_data[feature_cols].copy()
        
        feat_std = sample_data[feat].std()
        
        # Only perturb if feature has variance
        if feat_std == 0:
            sensitivity_ratios[feat] = {
                'sensitivity': 0.0,
                'is_proxy': feat in proxy_features
            }
            continue

        # Perturb the feature by a fraction of its standard deviation
        perturbation_amount = feat_std * perturbation_strength
        perturbed_data[feat] += perturbation_amount

        new_probs = model_predict_proba(perturbed_data)
        
        # Calculate mean absolute change in prediction per unit of input change
        # Divided by perturbation_amount to get sensitivity 'per unit' change
        mean_sensitivity = np.abs(new_probs - base_probs).mean() / perturbation_strength

        sensitivity_ratios[feat] = {
            'sensitivity': mean_sensitivity,
            'is_proxy': feat in proxy_features
        }
    
    sens_df = pd.DataFrame(sensitivity_ratios).T
    sens_df.index.name = 'feature'
    sens_df = sens_df.reset_index()

    proxy_sensitivity = sens_df[sens_df['is_proxy']]['sensitivity'].mean()
    financial_sensitivity = sens_df[~sens_df['is_proxy']]['sensitivity'].mean()

    # Avoid division by zero for financial_sensitivity
    # Add a small epsilon to the denominator for numerical stability if financial sensitivity is near zero.
    lipschitz_ratio = proxy_sensitivity / max(financial_sensitivity, 1e-6)

    print("\n--- LIPSCHITZ FAIRNESS ANALYSIS ---")
    print("=" * 50)
    print(f"Avg sensitivity to proxy features: {proxy_sensitivity:.4f}")
    print(f"Avg sensitivity to financial features: {financial_sensitivity:.4f}")
    print(f"Proxy/Financial sensitivity ratio (R): {lipschitz_ratio:.3f}")

    if lipschitz_ratio < 0.5:
        print("PASS: Model is much more sensitive to financial features than proxies (R < 0.5).")
    elif lipschitz_ratio < 1.0:
        print("WARNING: Proxy sensitivity is comparable to financial sensitivity (0.5 <= R < 1.0). Requires attention.")
    else:
        print("FAIL: Model is MORE sensitive to proxies than financials (R >= 1.0). Requires urgent intervention.")
        
    return sens_df, lipschitz_ratio

# Execute Lipschitz fairness measurement
sensitivity_df, lipschitz_ratio_value = lipschitz_fairness(model_predict_proba, X_test, feature_cols, proxy_features)
```

**Output Explanation and Real-World Impact:**

The Lipschitz Fairness Analysis provides a macro view of our model's fairness beyond specific twin pairs. The calculated Proxy/Financial Sensitivity Ratio (R) is a crucial indicator for CreditGuard Financial.
-   If `R < 0.5`, it implies the model correctly prioritizes legitimate financial factors.
-   If `0.5 <= R < 1.0`, it signals that proxy features have a surprisingly comparable influence, warranting further investigation.
-   If `R >= 1.0`, it's a critical failure, indicating that the model is *more* sensitive to proxy features than to core financial attributes.

This metric helps me, as a Risk Manager, assess the fundamental robustness of the model against subtle biases. A high ratio suggests that even small, seemingly innocuous changes in proxy features can dramatically swing credit decisions, posing a significant ethical and regulatory risk. This might necessitate a re-evaluation of model architecture or feature engineering to reduce undue influence from proxy features.

```python
# --- Visualization: Lipschitz Sensitivity Bar Chart ---
plt.figure(figsize=(12, 7))
sns.barplot(x='sensitivity', y='feature', data=sensitivity_df.sort_values('sensitivity', ascending=False),
            palette=['red' if f in proxy_features else 'blue' for f in sensitivity_df.sort_values('sensitivity', ascending=False)['feature']])
plt.title('Feature Sensitivity to Prediction Changes (Lipschitz Analysis)', fontsize=14)
plt.xlabel('Average Prediction Sensitivity (per unit feature change)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', alpha=0.75)
plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Proxy Feature'),
                    plt.Line2D([0], [0], color='blue', lw=4, label='Financial Feature')],
           title='Feature Type')
plt.show()
```

The Lipschitz Sensitivity Bar Chart clearly ranks features by their influence on prediction changes. By color-coding proxy features (red) and financial features (blue), I can visually identify if proxy features exhibit unexpectedly high sensitivity. This visualization makes it easy to communicate complex fairness insights to CreditGuard Financial's executive team and model developers, highlighting which features, if perturbed slightly, would cause the largest shifts in credit decisions.

---

### 6. Task 5: Conceptual Understanding of Adversarial Debiasing (Mitigation Strategy)

While our audit focuses on *detecting* and *verifying* individual fairness, it's essential for me as a Risk Manager to also understand *mitigation* strategies. **Adversarial Debiasing** is a powerful, albeit complex, technique for mitigating bias directly during model training. It aims to build a model whose internal representations do not encode information about protected attributes, even indirectly through proxy features.

The core idea involves training two competing neural networks:
1.  **Predictor (↑):** A main model that predicts the target variable (e.g., credit default).
2.  **Adversary (↓):** A separate model that tries to predict the protected attribute (e.g., race, gender, or proxy for them like `ZIP_code`) from the *intermediate representations* learned by the predictor.

The predictor is trained to minimize its prediction error *while simultaneously* trying to "fool" the adversary. This is achieved using a **gradient reversal layer** that inverts the gradients from the adversary, effectively forcing the predictor's encoder to learn representations that are *independent* of the protected attribute. If the adversary cannot predict the protected attribute from the predictor's internal representations, then those representations (and thus the final predictions) are considered 'fair' with respect to that attribute.

**Mathematical Concept: Gradient Reversal Layer**

The gradient reversal layer (GRL) is a crucial component in adversarial debiasing. During the forward pass, it acts as an identity function, simply passing its input unchanged:
$$f_{GRL}(x) = x$$
However, during the backward pass (gradient calculation), it multiplies the gradient by a negative constant $\lambda$:
$$\frac{\partial L}{\partial x} = -\lambda \frac{\partial L}{\partial f_{GRL}(x)}$$
This effectively reverses the direction of the gradient flow for the adversary's loss, making the feature extractor (encoder) learn features that confuse the adversary while still being useful for the main predictor.

This conceptual understanding is vital for CreditGuard Financial to consider advanced bias mitigation techniques for future model development, especially when high individual fairness is a critical requirement.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Visualization: Adversarial Debiasing Architecture Diagram ---
# This will be a conceptual diagram, not executable code for the diagram itself.
print("\n--- CONCEPTUAL ADVERSARIAL DEBIASING ARCHITECTURE ---")
print("=" * 60)
print("This diagram illustrates the conceptual architecture for adversarial debiasing.")
print("The goal is for the 'Encoder' to learn representations that are useful for the 'Predictor'")
print("BUT simultaneously 'fool' the 'Adversary' into not being able to predict the protected attribute.")
print("\n" + "="*60)
print("Encoder (Feature Extractor) -> ( Predictor (Credit Default) ↑ , Adversary (Protected Attribute) ↓ )")
print("                          ^                                                        |")
print("                          |----- Gradient Reversal Layer ---------------------------|")
print("                          |")
print("                     Input Features")
print("\n" + "="*60)


class AdversarialDebiaser(nn.Module):
    """
    Conceptual adversarial debiasing architecture.
    This class defines the network structure but does not implement the training loop.

    - Predictor: predicts credit default from features.
    - Adversary: predicts a protected attribute from the predictor's
                 intermediate representation (output of the encoder).

    Training Goal:
    - Minimize predictor loss (accurate credit prediction).
    - Maximize adversary loss (confuse the adversary) by using a gradient reversal layer
      such that the encoder learns representations independent of the protected attribute.
    """
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()

        # Shared representation (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Predictor head (credit default)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # For binary classification (credit approval probability)
        )

        # Adversary head (protected attribute prediction, e.g., ZIP code group)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # For binary classification (e.g., high/low ZIP code group)
        )
        
        # Placeholder for Gradient Reversal Layer (GRL)
        # In a real implementation, this would be a custom autograd function.
        # Here we conceptually describe its effect.
        self.grl = lambda x: x # Forward pass is identity

    def forward(self, x):
        # Forward pass through the encoder
        representation = self.encoder(x)

        # Predictor uses the learned representation
        prediction = self.predictor(representation)

        # Adversary uses the learned representation, but its gradients are reversed
        # .detach() here conceptually represents the target for adversary as a fixed representation
        # and in the backward pass, GRL would reverse the gradients to the encoder.
        adversary_pred = self.adversary(self.grl(representation.detach())) # .detach() for conceptual separation

        return prediction, adversary_pred, representation

# Conceptual demonstration of the model instantiation and forward pass
# (No actual training will happen here as it's a conceptual section)
n_model_features = len(feature_cols) # Example number of features
hidden_layer_dim = 64

# Create a dummy AdversarialDebiaser instance
conceptual_debiaser_model = AdversarialDebiaser(n_model_features, hidden_layer_dim)
print(f"\nConceptual AdversarialDebiaser model instantiated with {n_model_features} features and {hidden_layer_dim} hidden dimension.")
print("Model Architecture:")
print(conceptual_debiaser_model)

# Conceptual forward pass (using a dummy input)
dummy_input = torch.randn(1, n_model_features) # Single dummy input
pred, adv_pred, rep = conceptual_debiaser_model(dummy_input)

print(f"\nConceptual forward pass output for a dummy input:")
print(f"  Predictor Output (Credit Probability): {pred.item():.4f}")
print(f"  Adversary Output (Protected Attribute Probability): {adv_pred.item():.4f}")
print(f"  Learned Representation shape: {rep.shape}")

print("\nAdversarial debiasing trains two competing objectives:")
print("  Predictor: minimize credit default prediction error")
print("  Adversary: predict protected attribute from representation")
print("  Encoder: fool the adversary while keeping predictor accurate")
print("\nIf the adversary CANNOT predict group membership from the")
print("learned representation, the model cannot discriminate with respect to that attribute.")

```

**Output Explanation and Real-World Impact:**

The conceptual architecture diagram and the `AdversarialDebiaser` class visually and programmatically illustrate how this advanced technique works. As a Risk Manager, understanding adversarial debiasing's mechanism, particularly the role of the gradient reversal layer, allows me to appreciate its potential to build "fair by design" models.

This method forces the model to learn intermediate representations that are disentangled from protected attributes, fundamentally reducing the risk of indirect bias. However, it's a trade-off: aggressively removing all protected attribute information might sometimes reduce predictive accuracy. For CreditGuard Financial, this technique is most appropriate when regulatory risk is severe, and the highest standards of individual fairness are paramount, even at a slight cost to overall model performance. It informs future strategic decisions on model development and ethical AI governance.

---

### 7. Task 6: Synthesizing Fairness Audit Findings

The final step in my role as a Risk Manager is to compile all findings from our comprehensive fairness audit into a concise synthesis report. This report is crucial for providing a holistic view of the model's fairness posture across detection, mitigation (conceptualized here), and individual verification phases. It summarizes key metrics and an overall assessment for CreditGuard Financial's stakeholders, including compliance, legal, and executive leadership.

This synthesis report helps CreditGuard Financial to:
-   **Consolidate evidence:** Present a unified picture of fairness performance.
-   **Inform strategy:** Guide decisions on model adjustments, policy changes, or further mitigation efforts.
-   **Demonstrate commitment:** Provide clear documentation of our rigorous fairness testing process.

To provide a complete picture for the synthesis report, we will use placeholder values for `group_metrics` and `mitigation_results` from prior (conceptual) group fairness analyses (D4-T2-C1 and D4-T2-C2). This demonstrates the comprehensive nature of the full fairness lifecycle.

```python
def topic2_synthesis_report(group_metrics, mitigation_results, individual_violations_df, lipschitz_ratio):
    """
    Compiles a comprehensive fairness assessment across detection, mitigation,
    and individual verification phases.

    Args:
        group_metrics (dict): Placeholder for group fairness detection results (from D4-T2-C1).
        mitigation_results (dict): Placeholder for group fairness mitigation results (from D4-T2-C2).
        individual_violations_df (pd.DataFrame): DataFrame of individual violations (from Task 2).
        lipschitz_ratio (float): The Proxy/Financial sensitivity ratio (from Task 4).

    Returns:
        dict: A dictionary containing the full synthesis report.
    """
    
    # Calculate key individual fairness metrics from individual_violations_df
    num_twin_pairs_tested = len(individual_violations_df)
    decision_flips = individual_violations_df['decision_flipped'].sum()
    flip_rate = decision_flips / num_twin_pairs_tested if num_twin_pairs_tested > 0 else 0.0
    
    # Determine overall assessment based on defined thresholds
    # Individual Fairness Criteria: Decision Flip Rate < 0.10 AND Lipschitz Ratio < 1.0
    individual_fair_assessment = "INDIVIDUALLY FAIR" if (flip_rate < 0.10 and lipschitz_ratio < 1.0) else "INDIVIDUAL FAIRNESS CONCERNS"

    report = {
        'title': 'COMPREHENSIVE AI FAIRNESS ASSESSMENT',
        'scope': 'Credit Default Model XGBoost v2.1',
        'date': pd.Timestamp.now().isoformat(),
        
        'phase_1_group_detection': {
            'case': 'D4-T2-C1 (Conceptual)',
            'finding': f"DIR = {group_metrics['dir']:.3f} ({'PASS' if group_metrics.get('four_fifths_rule_pass', False) else 'FAIL'})",
            'proxies_detected': group_metrics.get('proxies_detected', 'N/A'),
            'details': "Conceptual metrics: D4-T2-C1 focused on detecting group disparities and identifying proxy features."
        },
        
        'phase_2_group_mitigation': {
            'case': 'D4-T2-C2 (Conceptual)',
            'strategy_applied': mitigation_results.get('strategy_applied', 'N/A'),
            'post_mitigation_dir': mitigation_results.get('post_mitigation_dir', 'N/A'),
            'auc_cost': mitigation_results.get('auc_cost', 'N/A'),
            'details': "Conceptual metrics: D4-T2-C2 focused on mitigating group-level biases using reweighting/constraints."
        },
        
        'phase_3_individual_verification': {
            'case': 'D4-T2-C3 (Individual Fairness Audit)',
            'twin_pairs_tested': num_twin_pairs_tested,
            'decision_flips': decision_flips,
            'flip_rate': f"{flip_rate:.1%}",
            'lipschitz_ratio': f"{lipschitz_ratio:.3f}",
            'individual_fairness_assessment': individual_fair_assessment
        },
        
        'overall_assessment': 'CONDITIONALLY FAIR' if (individual_fair_assessment == "INDIVIDUALLY FAIR" and group_metrics.get('four_fifths_rule_pass', False)) else 'REQUIRES FURTHER MITIGATION',
        
        'remaining_actions': [
            'Monitor twin pair flip rate monthly/quarterly',
            'Investigate top proxy features with legal counsel for potential disparate impact',
            'Review and enhance adverse action explanation pipeline',
            'Annual full fairness reaudit required'
        ]
    }
    
    print("--- COMPREHENSIVE AI FAIRNESS ASSESSMENT ---")
    print("(Topics 2 Synthesis: D4-T2-C1 + C2 + C3)")
    print("=" * 60)
    for phase, content in report.items():
        if phase.startswith('phase_'):
            print(f"\n{phase.replace('_', ' ').upper()}:")
            for k, v in content.items():
                print(f"  {k}: {v}")
        elif phase in ['title', 'scope', 'date']:
            print(f"{phase.replace('_', ' ').title()}: {content}")
    
    print(f"\nOVERALL ASSESSMENT: {report['overall_assessment']}")
    
    print(f"\nREMAINING ACTIONS:")
    for action in report['remaining_actions']:
        print(f"- {action}")
        
    return report

# --- Placeholder/conceptual inputs for Group Fairness from prior modules ---
# In a full workflow, these would be actual results from D4-T2-C1 and D4-T2-C2.
conceptual_group_metrics = {
    'dir': 0.82, # Disparate Impact Ratio
    'four_fifths_rule_pass': True, # Assume it passed group fairness for this example
    'proxies_detected': 3
}

conceptual_mitigation_results = {
    'strategy_applied': 'Reweighting + Fairness Constraints (Conceptual)',
    'post_mitigation_dir': 0.88,
    'auc_cost': -0.015 # Small AUC reduction due to mitigation
}

# Execute synthesis report generation
synthesis_report = topic2_synthesis_report(
    conceptual_group_metrics,
    conceptual_mitigation_results,
    violations_df,
    lipschitz_ratio_value
)
```

**Output Explanation and Real-World Impact:**

The `COMPREHENSIVE AI FAIRNESS ASSESSMENT` provides a dashboard for CreditGuard Financial to quickly grasp the model's fairness profile. It integrates findings from group-level analysis (conceptualized as D4-T2-C1 and D4-T2-C2) with the individual-level audit conducted in this notebook.

The report clearly states the `Decision Flip Rate` and `Lipschitz Ratio`, along with an `individual_fairness_assessment`. If the `overall_assessment` is "REQUIRES FURTHER MITIGATION", it immediately signals to the executive team that despite passing group-level tests, the model exhibits concerning individual-level biases. The `remaining_actions` section then provides concrete, actionable steps for CreditGuard Financial's compliance and development teams. This synthesis is the ultimate deliverable, guiding strategic decisions on responsible AI and demonstrating our commitment to fair dealing in accordance with CFA Standard III(B).

```python
# --- Visualization: Topic 2 Synthesis Dashboard ---
# This dashboard aggregates key metrics from all phases.

plt.figure(figsize=(14, 8))
gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

# Phase 1: Group Detection Summary
ax0 = plt.subplot(gs[0, 0])
ax0.text(0.5, 0.7, f"Phase 1: Group Detection (D4-T2-C1)", ha='center', va='center', fontsize=14, weight='bold')
ax0.text(0.5, 0.4, f"DIR: {conceptual_group_metrics['dir']:.3f} ({'PASS' if conceptual_group_metrics['four_fifths_rule_pass'] else 'FAIL'})", ha='center', va='center', fontsize=12)
ax0.text(0.5, 0.1, f"Proxies Detected: {conceptual_group_metrics['proxies_detected']}", ha='center', va='center', fontsize=12)
ax0.set_title('Group Fairness Detection', fontsize=16, pad=15)
ax0.axis('off')

# Phase 2: Group Mitigation Summary
ax1 = plt.subplot(gs[0, 1])
ax1.text(0.5, 0.7, f"Phase 2: Group Mitigation (D4-T2-C2)", ha='center', va='center', fontsize=14, weight='bold')
ax1.text(0.5, 0.4, f"Strategy: {conceptual_mitigation_results['strategy_applied']}", ha='center', va='center', fontsize=12)
ax1.text(0.5, 0.1, f"Post-Mitigation DIR: {conceptual_mitigation_results['post_mitigation_dir']:.3f} | AUC Cost: {conceptual_mitigation_results['auc_cost']:.2%}", ha='center', va='center', fontsize=12)
ax1.set_title('Group Fairness Mitigation (Conceptual)', fontsize=16, pad=15)
ax1.axis('off')

# Phase 3: Individual Verification Summary
ax2 = plt.subplot(gs[1, :])
ax2.text(0.5, 0.85, f"Phase 3: Individual Verification (D4-T2-C3)", ha='center', va='center', fontsize=14, weight='bold')
ax2.text(0.2, 0.6, f"Twin Pairs Tested: {len(violations_df)}", ha='center', va='center', fontsize=12)
ax2.text(0.8, 0.6, f"Decision Flips: {violations_df['decision_flipped'].sum()} ({violations_df['decision_flipped'].mean():.1%})", ha='center', va='center', fontsize=12)
ax2.text(0.2, 0.35, f"Materiality Violations: {violations_df['is_violation'].sum()} ({violations_df['is_violation'].mean():.1%})", ha='center', va='center', fontsize=12)
ax2.text(0.8, 0.35, f"Lipschitz Ratio (R): {lipschitz_ratio_value:.3f}", ha='center', va='center', fontsize=12)
ax2.text(0.5, 0.1, f"Individual Fairness: {synthesis_report['phase_3_individual_verification']['individual_fairness_assessment']}", ha='center', va='center', fontsize=14, color='red' if synthesis_report['phase_3_individual_verification']['individual_fairness_assessment'] != "INDIVIDUALLY FAIR" else 'green', weight='bold')
ax2.set_title('Individual Fairness Verification', fontsize=16, pad=15)
ax2.axis('off')

# Overall Assessment and Key Actions
ax3 = plt.subplot(gs[2, :])
overall_status_color = 'red' if 'REQUIRES FURTHER MITIGATION' in synthesis_report['overall_assessment'] else 'green'
ax3.text(0.5, 0.8, f"Overall Assessment: {synthesis_report['overall_assessment']}", ha='center', va='center', fontsize=18, color=overall_status_color, weight='bold')
actions_text = "\n".join([f"- {action}" for action in synthesis_report['remaining_actions']])
ax3.text(0.05, 0.5, "Key Remaining Actions:", ha='left', va='top', fontsize=14, weight='bold')
ax3.text(0.05, 0.05, actions_text, ha='left', va='top', fontsize=12)
ax3.set_title('Comprehensive Fairness Status', fontsize=16, pad=15)
ax3.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Topic 2: Comprehensive AI Fairness Audit Dashboard', fontsize=20, weight='bold', y=1.0)
plt.show()
```

The Topic 2 Synthesis Dashboard visually consolidates all the key metrics and assessments. This dashboard serves as a vital communication tool for CreditGuard Financial, allowing executives and compliance officers to quickly understand the model's fairness posture across group detection, group mitigation (conceptual), and individual verification. The prominent "Overall Assessment" and "Key Remaining Actions" sections clearly articulate the status and next steps, directly supporting my role as a Risk Manager in guiding ethical AI deployment.

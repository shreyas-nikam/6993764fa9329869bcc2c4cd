import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import io
import base64
import warnings

# Suppress XGBoost warning about use_label_encoder
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

# --- Global Constants / Configuration ---
N_SAMPLES = 5000
N_AUDIT_PAIRS = 100
RANDOM_SEED = 42
DECISION_THRESHOLD = 0.5
MATERIALITY_THRESHOLD = 0.10
PERTURBATION_STRENGTH = 0.1


# --- 1. Data Simulation and Model Training Functions ---

def prepare_credit_data(n_samples: int = N_SAMPLES, random_state: int = RANDOM_SEED):
    """
    Simulates a credit loan applicant dataset, introduces correlations,
    encodes categorical features, and splits the data.

    Args:
        n_samples (int): Number of samples to generate.
        random_state (int): Seed for random number generation.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list, list, list, LabelEncoder)
            df: The full DataFrame.
            X_train: Training features.
            X_test: Testing features.
            y_train: Training target.
            y_test: Testing target.
            feature_cols: List of all feature column names used by the model.
            financial_features: List of financial feature names.
            proxy_features: List of proxy feature names.
            label_encoder_home_ownership: The LabelEncoder instance for 'home_ownership'.
    """
    np.random.seed(random_state)

    data = {
        'FICO_score': np.random.randint(600, 850, n_samples),
        'income': np.random.randint(40000, 150000, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.5, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'employment_length_years': np.random.randint(0, 20, n_samples),
        'revolving_utilization': np.random.uniform(0.1, 0.9, n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.3, 0.3, 0.4]),
        'ZIP_code': np.random.choice(['10001', '10025', '10456', '90210', '77001'], n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'default': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)

    # Introduce some correlation
    df.loc[df['ZIP_code'] == '10456', 'default'] = np.random.choice(
        [0, 1], sum(df['ZIP_code'] == '10456'), p=[0.4, 0.6])
    df.loc[df['revolving_utilization'] > 0.7, 'default'] = np.random.choice(
        [0, 1], sum(df['revolving_utilization'] > 0.7), p=[0.3, 0.7])

    # Encode categorical features
    le_home_ownership = LabelEncoder()
    df['home_ownership_encoded'] = le_home_ownership.fit_transform(
        df['home_ownership'])
    le_zip_code = LabelEncoder()  # Create a separate LE for ZIP for clarity
    df['ZIP_code_encoded'] = le_zip_code.fit_transform(df['ZIP_code'])

    # Define features and target
    financial_features = ['FICO_score', 'income',
                          'debt_to_income_ratio', 'loan_amount', 'employment_length_years']
    proxy_features = ['revolving_utilization',
                      'home_ownership_encoded', 'ZIP_code_encoded']
    feature_cols = financial_features + proxy_features
    target_col = 'default'

    X = df[feature_cols]
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    print("Setup complete: Dataset simulated and prepared.")
    print(f"Financial Features: {financial_features}")
    print(f"Proxy Features: {proxy_features}")
    print(f"Total Features used by model: {feature_cols}")

    return df, X_train, X_test, y_train, y_test, feature_cols, financial_features, proxy_features, le_home_ownership


def train_credit_scoring_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_SEED):
    """
    Trains an XGBoost classifier as a placeholder credit scoring model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Seed for random number generation.

    Returns:
        tuple: (xgb.XGBClassifier, callable)
            model: The trained XGBoost model.
            model_predict_proba: A lambda function that takes a DataFrame/NumPy array
                                 and returns prediction probabilities for class 1.
    """
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, random_state=random_state)
    model.fit(X_train, y_train)

    def model_predict_proba(x): return model.predict_proba(x)[:, 1]

    print("XGBoost model trained.")
    return model, model_predict_proba


# --- 2. Twin Applicant Construction Function ---

def construct_twin_pairs(X_data: pd.DataFrame, financial_features: list, proxy_features: list, n_pairs: int = N_AUDIT_PAIRS, seed: int = RANDOM_SEED):
    """
    Constructs 'twin applicant' pairs for individual fairness auditing.

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

    sample_indices = np.random.choice(
        len(X_data), min(n_pairs, len(X_data)), replace=False)

    for idx in sample_indices:
        original = X_data.iloc[idx].copy()
        twin = original.copy()

        perturbed_proxies = {}
        for proxy_feat in proxy_features:
            original_proxy_value = original[proxy_feat]
            col_values = X_data[proxy_feat]

            # Determine the "opposite" typical value for the proxy feature
            if original_proxy_value > col_values.median():
                opposite_group_values = col_values[col_values <= col_values.median(
                )]
                if not opposite_group_values.empty:
                    twin[proxy_feat] = opposite_group_values.median()
                else:
                    twin[proxy_feat] = col_values.min()
            else:
                opposite_group_values = col_values[col_values > col_values.median(
                )]
                if not opposite_group_values.empty:
                    twin[proxy_feat] = opposite_group_values.median()
                else:
                    twin[proxy_feat] = col_values.max()

            # Only record if value actually changed
            if original_proxy_value != twin[proxy_feat]:
                perturbed_proxies[proxy_feat] = (
                    original_proxy_value, twin[proxy_feat])

        pairs.append({
            'original': original,
            'twin': twin,
            'original_idx': idx,
            'proxy_changed': list(perturbed_proxies.keys())
        })

    print(f"Constructed {len(pairs)} twin applicant pairs.")
    print(f"Proxy features varied: {proxy_features}")
    print("All other features (financial_features) are IDENTICAL between twins.")
    return pairs


# --- 3. Individual Fairness Violation Detection Functions ---

def detect_individual_violations(
    model_predict_proba: callable,
    twin_pairs: list,
    feature_cols: list,
    decision_threshold: float = DECISION_THRESHOLD,
    materiality_threshold: float = MATERIALITY_THRESHOLD
):
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
    print(
        f"Prediction delta > {materiality_threshold} (Violation Rate): {n_violations} ({n_violations / len(viol_df):.1%})")
    print(
        f"Decision flipped (Decision Flip Rate): {n_flipped} ({n_flipped / len(viol_df):.1%})")
    print(f"Mean prediction delta: {viol_df['delta'].mean():.4f}")
    print(f"Max prediction delta: {viol_df['delta'].max():.4f}")

    print("\nWORST VIOLATIONS (Top 5 by delta):")
    worst_violations = viol_df.nlargest(5, 'delta')
    for _, row in worst_violations.iterrows():
        flip_flag = "<<< DECISION FLIPPED >>>" if row['decision_flipped'] else ""
        print(f"Pair {row['original_idx']}: P(orig) {row['orig_prob']:.3f} -> P(twin) {row['twin_prob']:.3f} (delta={row['delta']:.3f}) {flip_flag} Proxy features changed: {row['proxy_changed']}")

    return viol_df


def plot_prediction_delta_distribution(violations_df: pd.DataFrame, materiality_threshold: float = MATERIALITY_THRESHOLD):
    """
    Plots the distribution of prediction deltas across twin pairs.

    Args:
        violations_df (pd.DataFrame): DataFrame containing individual fairness violations.
        materiality_threshold (float): Threshold for prediction difference to be considered a 'violation'.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(violations_df['delta'], bins=30, kde=True, color='skyblue')
    plt.axvline(x=materiality_threshold, color='red', linestyle='--',
                label=f'Materiality Threshold ($\epsilon={materiality_threshold:.2f}$)')
    plt.axvspan(materiality_threshold, violations_df['delta'].max(
    ), color='red', alpha=0.1, label='Individual Fairness Violations')

    decision_flip_deltas = violations_df[violations_df['decision_flipped']]['delta']
    if not decision_flip_deltas.empty:
        min_flip_delta = decision_flip_deltas.min()
        plt.axvspan(min_flip_delta, violations_df['delta'].max(
        ), color='purple', alpha=0.15, label='Decision Flip Region')

    plt.title('Distribution of Prediction Deltas Across Twin Pairs', fontsize=14)
    plt.xlabel(
        'Absolute Prediction Probability Difference ($\Delta P$)', fontsize=12)
    plt.ylabel('Number of Twin Pairs', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()


# --- 4. SHAP Decomposition and Gallery Functions ---

def decompose_twin_difference(model: xgb.XGBClassifier, pair: dict, feature_cols: list):
    """
    Uses SHAP to decompose the prediction difference for a twin pair,
    identifying which features drive the divergence.

    Args:
        model (xgb.XGBClassifier): The trained credit scoring model.
        pair (dict): A dictionary containing 'original' and 'twin' applicant data.
        feature_cols (list): List of features (column names) used by the model.

    Returns:
        tuple: (pd.DataFrame, pd.Series)
            feature_impact: DataFrame detailing SHAP value differences per feature.
            top_driver: Series representing the feature with the largest absolute SHAP difference.
    """
    explainer = shap.TreeExplainer(model)

    original_features_df = pair['original'][feature_cols].to_frame().T
    twin_features_df = pair['twin'][feature_cols].to_frame().T

    orig_shap_values = explainer.shap_values(original_features_df)[0]
    twin_shap_values = explainer.shap_values(twin_features_df)[0]

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

    top_driver = feature_impact.iloc[0]

    return feature_impact, top_driver


def plot_shap_waterfall_comparison(model: xgb.XGBClassifier, worst_violation_pair: dict, feature_cols: list, worst_pair_row: pd.Series):
    """
    Generates SHAP waterfall plots for the original and twin applicant
    of the single worst individual fairness violation pair.

    Args:
        model (xgb.XGBClassifier): The trained credit scoring model.
        worst_violation_pair (dict): The twin pair dictionary for the worst violation.
        feature_cols (list): List of features (column names) used by the model.
        worst_pair_row (pd.Series): The row from violations_df corresponding to the worst pair.
    """
    print("\n--- SHAP Waterfall Comparisons for the SINGLE WORST VIOLATION PAIR ---")
    print("=" * 70)

    explainer = shap.TreeExplainer(model)
    worst_orig_df = worst_violation_pair['original'][feature_cols].to_frame().T
    worst_twin_df = worst_violation_pair['twin'][feature_cols].to_frame().T

    worst_original_shap_values = explainer.shap_values(worst_orig_df)[0]
    worst_twin_shap_values = explainer.shap_values(worst_twin_df)[0]
    expected_value = explainer.expected_value

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    plt.sca(axes[0])
    shap.waterfall_plot(shap.Explanation(values=worst_original_shap_values, base_values=expected_value,
                        data=worst_orig_df.iloc[0], feature_names=feature_cols), max_display=10, show=False)
    axes[0].set_title(
        f'Worst Original Applicant {worst_pair_row["original_idx"]} SHAP (P={worst_pair_row["orig_prob"]:.3f})', fontsize=14)
    axes[0].set_xlabel('SHAP Value (impact on model output)', fontsize=12)

    plt.sca(axes[1])
    shap.waterfall_plot(shap.Explanation(values=worst_twin_shap_values, base_values=expected_value,
                        data=worst_twin_df.iloc[0], feature_names=feature_cols), max_display=10, show=False)
    axes[1].set_title(
        f'Worst Twin Applicant SHAP (P={worst_pair_row["twin_prob"]:.3f})', fontsize=14)
    axes[1].set_xlabel('SHAP Value (impact on model output)', fontsize=12)

    plt.tight_layout()
    plt.suptitle(
        f'SHAP Waterfall Comparison for Worst Violation Pair {worst_pair_row["original_idx"]}', fontsize=16, y=1.02)
    plt.show()


# --- 5. Lipschitz Fairness Analysis Functions ---

def _as_1d_prob(x):
    """
    Normalize model output to a 1D numpy array of class-1 probabilities.
    Accepts: (n,), (n,1), (n,2) -> returns (n,)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        if x.shape[1] == 1:
            return x[:, 0]
        if x.shape[1] >= 2:
            return x[:, 1]
    raise ValueError(f"Unexpected probability output shape: {x.shape}")


def lipschitz_fairness(
    model_predict_proba: callable,
    X_data: pd.DataFrame,
    feature_cols: list,
    proxy_features: list,
    n_sample: int = 500,
    perturbation_strength: float = PERTURBATION_STRENGTH,
    seed: int = RANDOM_SEED
):
    """
    Measures model sensitivity to small perturbations in input features,
    differentiating between financial and proxy features to assess fairness.

    Args:
        model_predict_proba (callable): A function that takes a DataFrame or NumPy array
                                        of features and returns prediction probabilities (for class 1).
        X_data (pd.DataFrame): The dataset from which to sample applicants for sensitivity testing.
        feature_cols (list): List of all feature column names used by the model.
        proxy_features (list): List of column names representing potentially proxy-correlated features.
        n_sample (int): Number of samples to use for sensitivity calculation.
        perturbation_strength (float): The strength of perturbation as a multiplier of std dev.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (pd.DataFrame, float)
            sens_df: DataFrame summarizing sensitivity for each feature.
            lipschitz_ratio: Ratio of average proxy sensitivity to average financial sensitivity.
    """
    np.random.seed(seed)

    if not isinstance(X_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data, columns=feature_cols)

    sample_data = X_data.sample(
        min(n_sample, len(X_data)), random_state=seed).copy()
    base_probs = _as_1d_prob(model_predict_proba(sample_data[feature_cols]))
    sensitivity_rows = []

    for feat in feature_cols:
        perturbed_data = sample_data[feature_cols].copy()
        feat_std = float(sample_data[feat].std(ddof=0))
        if feat_std == 0 or np.isnan(feat_std):
            sensitivity_rows.append({
                "feature": feat,
                "sensitivity": 0.0,
                "is_proxy": feat in proxy_features
            })
            continue

        perturbation_amount = feat_std * perturbation_strength
        perturbed_data[feat] = perturbed_data[feat] + perturbation_amount
        new_probs = _as_1d_prob(model_predict_proba(perturbed_data))

        if new_probs.shape != base_probs.shape:
            raise ValueError(
                f"Probability shape mismatch for feature '{feat}': "
                f"base={base_probs.shape}, new={new_probs.shape}"
            )

        mean_sensitivity = (
            np.abs(new_probs - base_probs).mean()) / perturbation_amount
        sensitivity_rows.append({
            "feature": feat,
            "sensitivity": float(mean_sensitivity),
            "is_proxy": feat in proxy_features
        })

    sens_df = pd.DataFrame(sensitivity_rows)

    proxy_sensitivity = sens_df.loc[sens_df["is_proxy"], "sensitivity"].mean()
    financial_sensitivity = sens_df.loc[~sens_df["is_proxy"], "sensitivity"].mean(
    )

    proxy_sensitivity = 0.0 if pd.isna(
        proxy_sensitivity) else float(proxy_sensitivity)
    financial_sensitivity = 0.0 if pd.isna(
        financial_sensitivity) else float(financial_sensitivity)

    lipschitz_ratio = proxy_sensitivity / max(financial_sensitivity, 1e-6)

    print("\n--- LIPSCHITZ FAIRNESS ANALYSIS ---")
    print("=" * 50)
    print(f"Avg sensitivity to proxy features: {proxy_sensitivity:.4f}")
    print(
        f"Avg sensitivity to financial features: {financial_sensitivity:.4f}")
    print(f"Proxy/Financial sensitivity ratio (R): {lipschitz_ratio:.3f}")

    if lipschitz_ratio < 0.5:
        print("PASS: Model is much more sensitive to financial features than proxies (R < 0.5).")
    elif lipschitz_ratio < 1.0:
        print("WARNING: Proxy sensitivity is comparable to financial sensitivity (0.5 <= R < 1.0). Requires attention.")
    else:
        print("FAIL: Model is MORE sensitive to proxies than financials (R >= 1.0). Requires urgent intervention.")

    return sens_df.sort_values("sensitivity", ascending=False).reset_index(drop=True), lipschitz_ratio


def plot_lipschitz_sensitivity(sensitivity_df: pd.DataFrame, proxy_features: list):
    """
    Plots a bar chart of feature sensitivities from Lipschitz analysis.

    Args:
        sensitivity_df (pd.DataFrame): DataFrame containing feature sensitivities.
        proxy_features (list): List of proxy feature names.
    """
    plt.figure(figsize=(12, 7))
    sorted_df = sensitivity_df.sort_values('sensitivity', ascending=False)
    sns.barplot(x='sensitivity', y='feature', data=sorted_df,
                palette=['red' if f in proxy_features else 'blue' for f in sorted_df['feature']])
    plt.title(
        'Feature Sensitivity to Prediction Changes (Lipschitz Analysis)', fontsize=14)
    plt.xlabel(
        'Average Prediction Sensitivity (per unit feature change)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', alpha=0.75)
    plt.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Proxy Feature'),
                        plt.Line2D([0], [0], color='blue', lw=4, label='Financial Feature')],
               title='Feature Type')
    plt.show()


# --- 6. Adversarial Debiasing (Conceptual) Functions ---

class AdversarialDebiaser:
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

    def __init__(self, n_features: int, hidden_dim: int = 64):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.encoder_layers = [
            f"Linear(in_features={n_features}, out_features={hidden_dim})",
            "ReLU()",
            f"Linear(in_features={hidden_dim}, out_features={hidden_dim // 2})",
            "ReLU()"
        ]
        self.predictor_layers = [
            f"Linear(in_features={hidden_dim // 2}, out_features=1)",
            "Sigmoid()"
        ]
        self.adversary_layers = [
            f"Linear(in_features={hidden_dim // 2}, out_features=1)",
            "Sigmoid()"
        ]

    def __str__(self):
        return (
            f"AdversarialDebiaser(\n"
            f"  (encoder): Sequential(\n"
            f"    {chr(10).join('    ' + layer for layer in self.encoder_layers)}\n"
            f"  )\n"
            f"  (predictor): Sequential(\n"
            f"    {chr(10).join('    ' + layer for layer in self.predictor_layers)}\n"
            f"  )\n"
            f"  (adversary): Sequential(\n"
            f"    {chr(10).join('    ' + layer for layer in self.adversary_layers)}\n"
            f"  )\n"
            f"  (grl): GradientReversalLayer()\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()


def display_adversarial_debiasing_concept(feature_cols: list):
    """
    Prints a conceptual diagram and instantiates a dummy AdversarialDebiaser model
    to illustrate its architecture.

    Args:
        feature_cols (list): List of feature names to determine input dimension.
    """
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

    n_model_features = len(feature_cols)
    hidden_layer_dim = 64

    conceptual_debiaser_model = AdversarialDebiaser(
        n_model_features, hidden_layer_dim)
    print(
        f"\nConceptual AdversarialDebiaser model instantiated with {n_model_features} features and {hidden_layer_dim} hidden dimension.")
    print("Model Architecture:")
    print(conceptual_debiaser_model)

    print(f"\nConceptual architecture components:")
    print(f"  Input: {n_model_features} features")
    print(
        f"  Encoder output: {hidden_layer_dim // 2} dimensional representation")
    print(f"  Predictor output: 1 value (credit default probability)")
    print(f"  Adversary output: 1 value (protected attribute probability)")

    print("\nAdversarial debiasing trains two competing objectives:")
    print("  Predictor: minimize credit default prediction error")
    print("  Adversary: predict protected attribute from representation")
    print("  Encoder: fool the adversary while keeping predictor accurate")
    print("\nIf the adversary CANNOT predict group membership from the")
    print("learned representation, the model cannot discriminate with respect to that attribute.")


# --- 7. Synthesis Report Functions ---

def topic2_synthesis_report(group_metrics: dict, mitigation_results: dict, individual_violations_df: pd.DataFrame, lipschitz_ratio: float):
    """
    Compiles a comprehensive fairness assessment across detection, mitigation,
    and individual verification phases.

    Args:
        group_metrics (dict): Placeholder for group fairness detection results (e.g., from D4-T2-C1).
        mitigation_results (dict): Placeholder for group fairness mitigation results (e.g., from D4-T2-C2).
        individual_violations_df (pd.DataFrame): DataFrame of individual violations (from Task 2).
        lipschitz_ratio (float): The Proxy/Financial sensitivity ratio (from Task 4).

    Returns:
        dict: A dictionary containing the full synthesis report.
    """
    num_twin_pairs_tested = len(individual_violations_df)
    decision_flips = individual_violations_df['decision_flipped'].sum()
    flip_rate = decision_flips / \
        num_twin_pairs_tested if num_twin_pairs_tested > 0 else 0.0

    individual_fair_assessment = "INDIVIDUALLY FAIR" if (
        flip_rate < 0.10 and lipschitz_ratio < 1.0) else "INDIVIDUAL FAIRNESS CONCERNS"
    overall_assessment = 'CONDITIONALLY FAIR' if (individual_fair_assessment == "INDIVIDUALLY FAIR" and group_metrics.get(
        'four_fifths_rule_pass', False)) else 'REQUIRES FURTHER MITIGATION'

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

        'overall_assessment': overall_assessment,

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
        if isinstance(content, dict) and phase.startswith('phase_'):
            print(f"\n{phase.replace('_', ' ').upper()}:")
            for k, v in content.items():
                print(f"  {k}: {v}")
        elif phase in ['title', 'scope', 'date']:
            print(f"{phase.replace('_', ' ').title()}: {content}")
        elif phase == 'overall_assessment':
            print(f"\nOVERALL ASSESSMENT: {content}")
        elif phase == 'remaining_actions':
            print(f"\nREMAINING ACTIONS:")
            for action in content:
                print(f"- {action}")

    return report


def plot_synthesis_dashboard(synthesis_report: dict, violations_df: pd.DataFrame, lipschitz_ratio_value: float):
    """
    Generates a dashboard aggregating key metrics from all fairness phases.

    Args:
        synthesis_report (dict): The complete synthesis report.
        violations_df (pd.DataFrame): DataFrame containing individual fairness violations.
        lipschitz_ratio_value (float): The Lipschitz fairness ratio.
    """
    plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

    # Phase 1: Group Detection Summary
    ax0 = plt.subplot(gs[0, 0])
    group_metrics = synthesis_report['phase_1_group_detection']
    ax0.text(0.5, 0.7, f"Phase 1: Group Detection (D4-T2-C1)",
             ha='center', va='center', fontsize=14, weight='bold')
    ax0.text(
        0.5, 0.4, f"Finding: {group_metrics['finding']}", ha='center', va='center', fontsize=12)
    ax0.text(
        0.5, 0.1, f"Proxies Detected: {group_metrics['proxies_detected']}", ha='center', va='center', fontsize=12)
    ax0.set_title('Group Fairness Detection', fontsize=16, pad=15)
    ax0.axis('off')

    # Phase 2: Group Mitigation Summary
    ax1 = plt.subplot(gs[0, 1])
    mitigation_results = synthesis_report['phase_2_group_mitigation']
    ax1.text(0.5, 0.7, f"Phase 2: Group Mitigation (D4-T2-C2)",
             ha='center', va='center', fontsize=14, weight='bold')
    ax1.text(
        0.5, 0.4, f"Strategy: {mitigation_results['strategy_applied']}", ha='center', va='center', fontsize=12)
    auc_cost_str = f"{mitigation_results['auc_cost']:.2%}" if isinstance(
        mitigation_results['auc_cost'], (int, float)) else str(mitigation_results['auc_cost'])
    ax1.text(
        0.5, 0.1, f"Post-Mitigation DIR: {mitigation_results['post_mitigation_dir']} | AUC Cost: {auc_cost_str}", ha='center', va='center', fontsize=12)
    ax1.set_title('Group Fairness Mitigation (Conceptual)',
                  fontsize=16, pad=15)
    ax1.axis('off')

    # Phase 3: Individual Verification Summary
    ax2 = plt.subplot(gs[1, :])
    individual_verification = synthesis_report['phase_3_individual_verification']
    ax2.text(0.5, 0.85, f"Phase 3: Individual Verification (D4-T2-C3)",
             ha='center', va='center', fontsize=14, weight='bold')
    ax2.text(
        0.2, 0.6, f"Twin Pairs Tested: {individual_verification['twin_pairs_tested']}", ha='center', va='center', fontsize=12)
    ax2.text(
        0.8, 0.6, f"Decision Flips: {individual_verification['decision_flips']} ({individual_verification['flip_rate']})", ha='center', va='center', fontsize=12)
    ax2.text(
        0.2, 0.35, f"Materiality Violations: {violations_df['is_violation'].sum()} ({violations_df['is_violation'].mean():.1%})", ha='center', va='center', fontsize=12)
    ax2.text(
        0.8, 0.35, f"Lipschitz Ratio (R): {lipschitz_ratio_value:.3f}", ha='center', va='center', fontsize=12)
    individual_fairness_assessment = individual_verification['individual_fairness_assessment']
    ax2.text(0.5, 0.1, f"Individual Fairness: {individual_fairness_assessment}", ha='center', va='center',
             fontsize=14, color='red' if individual_fairness_assessment != "INDIVIDUALLY FAIR" else 'green', weight='bold')
    ax2.set_title('Individual Fairness Verification', fontsize=16, pad=15)
    ax2.axis('off')

    # Overall Assessment and Key Actions
    ax3 = plt.subplot(gs[2, :])
    overall_status_color = 'red' if 'REQUIRES FURTHER MITIGATION' in synthesis_report[
        'overall_assessment'] else 'green'
    ax3.text(0.5, 0.8, f"Overall Assessment: {synthesis_report['overall_assessment']}",
             ha='center', va='center', fontsize=18, color=overall_status_color, weight='bold')
    actions_text = "\n".join(
        [f"- {action}" for action in synthesis_report['remaining_actions']])
    ax3.text(0.05, 0.5, "Key Remaining Actions:", ha='left',
             va='top', fontsize=14, weight='bold')
    ax3.text(0.05, 0.05, actions_text, ha='left', va='top', fontsize=12)
    ax3.set_title('Comprehensive Fairness Status', fontsize=16, pad=15)
    ax3.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Topic 2: Comprehensive AI Fairness Audit Dashboard',
                 fontsize=20, weight='bold', y=1.0)
    plt.show()


def main():
    """
    Orchestrates the entire fairness audit workflow.
    """
    print("Starting AI Fairness Audit Workflow...")

    # --- 1. Data Simulation and Model Training ---
    (df, X_train, X_test, y_train, y_test,
     feature_cols, financial_features, proxy_features, le_home_ownership) = prepare_credit_data(
        n_samples=N_SAMPLES, random_state=RANDOM_SEED
    )

    model, model_predict_proba = train_credit_scoring_model(
        X_train, y_train, random_state=RANDOM_SEED)

    # --- 2. Twin Applicant Construction ---
    twin_applicant_pairs = construct_twin_pairs(
        X_test, financial_features, proxy_features, n_pairs=N_AUDIT_PAIRS, seed=RANDOM_SEED
    )

    # Display a sample twin pair for verification
    print("\n--- Sample Twin Pair Verification ---")
    if twin_applicant_pairs:
        sample_pair = twin_applicant_pairs[0]
        print(f"Original Applicant (Index {sample_pair['original_idx']}):")
        print(sample_pair['original'].to_frame().T.to_string())
        print("Twin Applicant:")
        print(sample_pair['twin'].to_frame().T.to_string())
        print(
            f"Features intentionally changed for twin: {sample_pair['proxy_changed']}")
    else:
        print("No twin pairs constructed.")

    # --- 3. Individual Fairness Violation Detection ---
    violations_df = detect_individual_violations(
        model_predict_proba, twin_applicant_pairs, feature_cols,
        decision_threshold=DECISION_THRESHOLD, materiality_threshold=MATERIALITY_THRESHOLD
    )
    plot_prediction_delta_distribution(
        violations_df, materiality_threshold=MATERIALITY_THRESHOLD)

    # --- 4. SHAP Decomposition and Gallery ---
    if not violations_df.empty:
        print("\n--- SHAP Analysis for Worst Violations ---")
        print("Note: For interactive gallery, please use the Streamlit app.py")

        worst_pair_row = violations_df.nlargest(1, 'delta').iloc[0]
        worst_original_idx = worst_pair_row['original_idx']
        worst_pair_data = next(
            (p for p in twin_applicant_pairs if p['original_idx'] == worst_original_idx), None)

        if worst_pair_data:
            plot_shap_waterfall_comparison(
                model, worst_pair_data, feature_cols, worst_pair_row)
        else:
            print(
                f"Warning: Could not find worst twin pair for original_idx {worst_original_idx}.")
    else:
        print("No violations detected, skipping SHAP decomposition and gallery.")

    # --- 5. Lipschitz Fairness Analysis ---
    sensitivity_df, lipschitz_ratio_value = lipschitz_fairness(
        model_predict_proba, X_test, feature_cols, proxy_features,
        n_sample=500, perturbation_strength=PERTURBATION_STRENGTH, seed=RANDOM_SEED
    )
    plot_lipschitz_sensitivity(sensitivity_df, proxy_features)

    # --- 6. Adversarial Debiasing (Conceptual) ---
    display_adversarial_debiasing_concept(feature_cols)

    # --- 7. Synthesis Report ---
    # Placeholder/conceptual inputs for Group Fairness from prior modules
    conceptual_group_metrics = {
        'dir': 0.82,
        'four_fifths_rule_pass': True,
        'proxies_detected': 3
    }
    conceptual_mitigation_results = {
        'strategy_applied': 'Reweighting + Fairness Constraints (Conceptual)',
        'post_mitigation_dir': 0.88,
        'auc_cost': -0.015
    }

    synthesis_report = topic2_synthesis_report(
        conceptual_group_metrics,
        conceptual_mitigation_results,
        violations_df,
        lipschitz_ratio_value
    )
    plot_synthesis_dashboard(
        synthesis_report, violations_df, lipschitz_ratio_value)

    print("\nAI Fairness Audit Workflow Completed.")


# --- Module-level initialization for Streamlit app ---
# NOTE: This module-level initialization has been disabled to prevent slow imports.
# The Streamlit app (app.py) now handles initialization using @st.cache_resource
# to cache the expensive model training and data preparation operations.
#
# If you want to run this as a standalone script, use: python source.py
# The initialization will happen inside main() function.

if __name__ == "__main__":
    main()


# Streamlit Application Specification: AI Fairness Audit

## 1. Application Overview

The **AI Fairness Audit: Twin Applicant Testing** application serves as a critical tool for CFA Charterholders and Risk Managers at CreditGuard Financial to rigorously audit AI-powered credit scoring models for individual-level fairness. While group fairness metrics are often met, this application addresses the critical gap of individual discrimination, where similar applicants receive different outcomes due to model reliance on proxy-correlated features.

The application guides the user through a comprehensive workflow:

1.  **Introduction & Setup**: Load the pre-trained credit model and applicant data, defining financial and proxy features, and setting key audit parameters.
2.  **Construct Twin Applicants**: Programmatically generate 'twin' applicant pairs by perturbing only the selected proxy features, keeping all financial features identical.
3.  **Detect Individual Violations**: Feed twin pairs into the model to identify 'decision flips' and 'material prediction deltas', quantifying individual fairness violation rates.
4.  **Decompose Unfairness (SHAP)**: Utilize SHAP explanations to pinpoint specific features (especially proxy features) driving the differing predictions for the worst-case unfair twin pairs.
5.  **Measure Model Sensitivity (Lipschitz)**: Quantify the model's overall sensitivity to proxy versus financial features using Lipschitz fairness principles, yielding a 'proxy/financial sensitivity ratio'.
6.  **Adversarial Debiasing (Conceptual)**: Provide a conceptual understanding and architectural diagram of adversarial debiasing as an advanced mitigation strategy.
7.  **Synthesize Fairness Audit Findings**: Compile all audit findings into a comprehensive report and dashboard, integrating individual-level results with conceptual group-level fairness metrics to provide actionable insights for compliance and risk management.

This structured workflow ensures a thorough, evidence-based assessment of individual fairness, allowing CreditGuard Financial to proactively mitigate regulatory risk and uphold ethical AI practices.

## 2. Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap # Required for waterfall plots
import io # For image buffer handling in Streamlit
import base64 # For encoding images if necessary (used by source.py's gallery function logic)

# Import all functions and pre-initialized objects from source.py
from source import (
    model, X_test, y_test, financial_features, proxy_features, feature_cols,
    model_predict_proba, construct_twin_pairs, detect_individual_violations,
    decompose_twin_difference, create_twin_applicant_gallery, # create_twin_applicant_gallery will be adapted for Streamlit display
    lipschitz_fairness, topic2_synthesis_report,
    conceptual_group_metrics, conceptual_mitigation_results,
    AdversarialDebiaser # For conceptual architecture representation
)
```

### `st.session_state` Design

The `st.session_state` will be initialized once upon the first run of the application and updated as the user interacts with widgets and generates results.

**Initialization:**

```python
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model = model # Pre-trained model
    st.session_state.X_test = X_test # Test dataset
    st.session_state.y_test = y_test # Test target variable
    st.session_state.financial_features = financial_features # List of financial features
    st.session_state.proxy_features_initial = proxy_features # Initial list of proxy features from source.py
    st.session_state.feature_cols = feature_cols # All features used by the model
    st.session_state.model_predict_proba = model_predict_proba # Model's predict_proba function

    # User-configurable parameters (with defaults)
    st.session_state.n_audit_pairs = 100
    st.session_state.materiality_threshold = 0.10
    st.session_state.selected_proxy_features = proxy_features # Default to initial proxy features, user can change

    # Results from various application steps
    st.session_state.twin_applicant_pairs = None # Result of construct_twin_pairs
    st.session_state.violations_df = None # Result of detect_individual_violations
    st.session_state.lipschitz_ratio_value = None # Result of lipschitz_fairness
    st.session_state.sensitivity_df = None # Result of lipschitz_fairness
    st.session_state.synthesis_report = None # Result of topic2_synthesis_report

    # Conceptual group fairness metrics (placeholders from source.py)
    st.session_state.conceptual_group_metrics = conceptual_group_metrics
    st.session_state.conceptual_mitigation_results = conceptual_mitigation_results
```

**`st.session_state` Keys and Usage:**

*   `initialized` (bool): Ensures one-time initialization of session state.
*   `current_page` (str): Stores the currently selected page from the sidebar for conditional rendering.
*   `model` (`xgboost.XGBClassifier`): Loaded model, read-only.
*   `X_test` (`pd.DataFrame`): Loaded test data, read-only.
*   `y_test` (`pd.Series`): Loaded test labels, read-only.
*   `financial_features` (list): List of financial features, read-only.
*   `proxy_features_initial` (list): Initial list of proxy features from `source.py`, read-only, used as default for `selected_proxy_features`.
*   `feature_cols` (list): All features used by the model, read-only.
*   `model_predict_proba` (callable): Model's prediction function, read-only.
*   `n_audit_pairs` (int): **Updated** by a `st.slider` on "1. Introduction & Setup". **Read** by `construct_twin_pairs`.
*   `materiality_threshold` (float): **Updated** by a `st.slider` on "3. Detect Individual Violations". **Read** by `detect_individual_violations`.
*   `selected_proxy_features` (list): **Updated** by `st.multiselect` on "1. Introduction & Setup". **Read** by `construct_twin_pairs` and `lipschitz_fairness` (for coloring).
*   `twin_applicant_pairs` (list of dicts): **Updated** by `construct_twin_pairs` on "2. Construct Twin Applicants". **Read** by `detect_individual_violations` and `decompose_twin_difference` (for SHAP).
*   `violations_df` (`pd.DataFrame`): **Updated** by `detect_individual_violations` on "3. Detect Individual Violations". **Read** by SHAP decomposition, Lipschitz fairness, and synthesis report.
*   `lipschitz_ratio_value` (float): **Updated** by `lipschitz_fairness` on "5. Measure Model Sensitivity (Lipschitz)". **Read** by synthesis report.
*   `sensitivity_df` (`pd.DataFrame`): **Updated** by `lipschitz_fairness` on "5. Measure Model Sensitivity (Lipschitz)". **Read** by Lipschitz Sensitivity Bar Chart.
*   `synthesis_report` (dict): **Updated** by `topic2_synthesis_report` on "7. Synthesize Fairness Audit Findings". **Read** by the synthesis report display and dashboard.
*   `conceptual_group_metrics` (dict): Placeholder group metrics, read-only.
*   `conceptual_mitigation_results` (dict): Placeholder mitigation results, read-only.

### Application Structure and Flow

```python
# app.py

# --- Imports and Session State Initialization (as defined above) ---

# Helper function to display plots and manage memory
def display_plot(fig):
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

# --- Application Sidebar and Page Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Choose a section",
    [
        "1. Introduction & Setup",
        "2. Construct Twin Applicants",
        "3. Detect Individual Violations",
        "4. Decompose Unfairness (SHAP)",
        "5. Measure Model Sensitivity (Lipschitz)",
        "6. Adversarial Debiasing (Conceptual)",
        "7. Synthesize Fairness Audit Findings"
    ]
)
st.session_state.current_page = page_selection

# --- Main Content Area ---
st.title("AI Fairness Audit: Twin Applicant Testing")
st.markdown(f"**Persona:** {'CFA Charterholder and Risk Manager'}")

# --- Page 1: Introduction & Setup ---
if st.session_state.current_page == "1. Introduction & Setup":
    st.header("1. Introduction & Setup: Model & Data Loading")
    st.markdown(f"""
    As a **CFA Charterholder and Risk Manager** at **CreditGuard Financial**, a leading institution committed to ethical AI practices, my primary responsibility is to ensure that our automated credit scoring models operate with the highest standards of fairness and compliance. Regulators, including those enforcing the Equal Credit Opportunity Act (ECOA), are increasingly scrutinizing AI models for individual-level discrimination, not just group disparities. My team recently deployed an AI-powered credit model, and while it has demonstrated strong group-level fairness metrics, I need to proactively audit it for individual-level unfairness – instances where similar individuals receive different outcomes.

    This application documents my workflow to conduct a rigorous individual fairness audit using 'twin applicant' testing, SHAP explanations, and Lipschitz fairness measurements. My goal is to identify and quantify any potential individual discrimination, understand its drivers, and assess the model's overall sensitivity to non-financial features. This proactive approach helps **CreditGuard Financial** mitigate regulatory risk, enhance our ethical reputation, and improve model robustness.

    ### Initial Setup: Installing Libraries and Loading Data

    To begin our individual fairness audit, we need to set up our environment by installing the necessary Python libraries and loading the pre-trained credit scoring model along with the applicant dataset. This ensures we have all the tools and data ready for constructing twin applicants and evaluating model fairness.
    """)

    st.subheader("Model and Data Overview")
    st.markdown(f"The credit scoring model and dataset have been pre-loaded from `source.py`.")
    st.markdown(f"**Model Type:** `xgboost.XGBClassifier`")
    st.markdown(f"**Number of features:** `{len(st.session_state.feature_cols)}`")
    st.markdown(f"**Number of test samples:** `{len(st.session_state.X_test)}`")

    st.markdown(f"**Financial Features (used by model):**")
    st.dataframe(pd.DataFrame(st.session_state.financial_features, columns=['Feature Name']))

    st.markdown(f"**Proxy Features (used by model, can be varied for twins):**")
    st.dataframe(pd.DataFrame(st.session_state.proxy_features_initial, columns=['Feature Name']))

    st.subheader("Configuration for Twin Generation")
    st.session_state.selected_proxy_features = st.multiselect(
        "Select Proxy Features for Twin Generation (features to vary):",
        options=st.session_state.feature_cols,
        default=st.session_state.proxy_features_initial
    )

    st.session_state.n_audit_pairs = st.slider(
        "Number of Twin Pairs to Generate for Audit:",
        min_value=10, max_value=500, value=100, step=10
    )
    st.info(f"The model and data are now loaded. You can proceed to configure the audit parameters on this page and then construct twin applicants.")

# --- Page 2: Construct Twin Applicants ---
elif st.session_state.current_page == "2. Construct Twin Applicants":
    st.header("2. Constructing Twin Applicants for Fair Auditing")
    st.markdown(f"""
    My first step as a Risk Manager is to create 'twin applicant' pairs. This critical technique allows me to isolate the impact of potentially biased or proxy features on credit decisions. I construct hypothetical applicants who are identical in all financially relevant aspects but differ only in a single, non-financial, potentially proxy-correlated feature (like `ZIP_code` or `revolving_utilization`). By comparing their model predictions, I can directly observe if the model is treating similar individuals dissimilarly, which is a key indicator of individual unfairness.

    This process is akin to the "matched pair" testing regulators have used for decades, now scaled with AI. It helps CreditGuard Financial identify subtle biases that aggregate group fairness metrics might overlook.
    """)

    if st.button("Construct Twin Applicants"):
        if not st.session_state.selected_proxy_features:
            st.warning("Please select at least one proxy feature for twin generation in '1. Introduction & Setup'.")
        else:
            with st.spinner("Constructing twin applicant pairs..."):
                twin_applicant_pairs = construct_twin_pairs(
                    st.session_state.X_test,
                    st.session_state.financial_features,
                    st.session_state.selected_proxy_features,
                    n_pairs=st.session_state.n_audit_pairs
                )
                st.session_state.twin_applicant_pairs = twin_applicant_pairs
            st.success(f"Constructed {len(twin_applicant_pairs)} twin applicant pairs.")

            st.markdown(f"""
            After constructing `{st.session_state.n_audit_pairs}` twin applicants, I can see that the `{st.session_state.financial_features}` remain constant for both the original and twin applicants, while the `{st.session_state.selected_proxy_features}` have been modified. This deliberate perturbation allows us to isolate the model's sensitivity to these specific proxy features. The generated pairs are now ready to be fed into our credit scoring model to detect any differential treatment.
            """)

            st.subheader("Sample Twin Pair Verification")
            if st.session_state.twin_applicant_pairs:
                sample_pair = st.session_state.twin_applicant_pairs[0]
                st.markdown(f"Original Applicant (Index `{sample_pair['original_idx']}`):")
                st.dataframe(sample_pair['original'].to_frame().T)
                st.markdown(f"Twin Applicant:")
                st.dataframe(sample_pair['twin'].to_frame().T)
                st.markdown(f"Features intentionally changed for twin: `{', '.join(sample_pair['proxy_changed'])}`")
            else:
                st.info("No twin pairs constructed yet. Click the button above.")

# --- Page 3: Detect Individual Violations ---
elif st.session_state.current_page == "3. Detect Individual Violations":
    st.header("3. Detecting Individual Fairness Violations")
    st.markdown(f"""
    Now that I have my twin applicant pairs, my next critical step is to feed them into our credit scoring model and analyze its predictions. As a Risk Manager, I'm specifically looking for 'decision flips' – where one twin is approved and the other denied – or 'material prediction deltas' – a significant difference in approval probability, even if both are approved or denied. This directly quantifies individual unfairness.

    The **Individual Fairness Violation Rate** measures the proportion of twin pairs where the absolute difference in prediction probabilities exceeds a predefined materiality threshold $\epsilon$. The **Decision Flip Rate** is a more severe metric, indicating instances where the model's final binary decision (approve/deny) changes between identical twins.
    """)

    st.markdown(r"**Individual Fairness Violation Rate ($VR_{\text{individual}}$):**")
    st.markdown(r"$$VR_{\text{individual}} = \frac{{\|\{(x, x') : \|f(x) - f(x')\| > \epsilon, x \sim_{\text{financial}} x'\}\|}}{{\|\text{all twin pairs}\|}}$$")
    st.markdown(r"where $x$ and $x'$ are a twin pair of applicants; $x \sim_{\text{financial}} x'$ means $x$ and $x'$ have identical financial features but different proxy features; $f(x)$ and $f(x')$ are the model's predicted credit approval probabilities for $x$ and $x'$, respectively; $\epsilon$ is the materiality threshold for prediction difference (e.g., 0.10); and $|\cdot|$ denotes the count of elements in a set.")

    st.markdown(r"**Decision Flip Rate ($FR$):**")
    st.markdown(r"$$FR = \frac{{\|\{(x, x') : \hat{y}(x) \neq \hat{y}(x')\}\|}}{{\|\text{all twin pairs}\|}}$$")
    st.markdown(r"where $\hat{y}(x)$ and $\hat{y}(x')$ are the model's binarized (e.g., using a 0.5 threshold) credit approval decisions for $x$ and $x'$, respectively; and $\hat{y}(x) \neq \hat{y}(x')$ indicates a decision flip.")

    st.markdown(f"""
    By quantifying these rates, I can provide concrete evidence of individual unfairness to our compliance team and, if necessary, to regulators.
    """)

    if st.session_state.twin_applicant_pairs is None:
        st.warning("Please construct twin applicants first on the '2. Construct Twin Applicants' page.")
    else:
        st.session_state.materiality_threshold = st.slider(
            "Set Materiality Threshold ($\epsilon$) for Prediction Delta:",
            min_value=0.01, max_value=0.50, value=0.10, step=0.01
        )

        if st.button("Detect Violations"):
            with st.spinner("Detecting individual fairness violations..."):
                violations_df = detect_individual_violations(
                    st.session_state.model_predict_proba,
                    st.session_state.twin_applicant_pairs,
                    st.session_state.feature_cols,
                    materiality_threshold=st.session_state.materiality_threshold
                )
                st.session_state.violations_df = violations_df
            st.success("Violation detection complete!")

            if st.session_state.violations_df is not None:
                st.subheader("Summary of Individual Fairness Violations")
                n_violations = st.session_state.violations_df['is_violation'].sum()
                n_flipped = st.session_state.violations_df['decision_flipped'].sum()
                total_pairs = len(st.session_state.violations_df)

                st.info(f"**Twin pairs tested:** `{total_pairs}`")
                st.info(f"**Prediction delta > {st.session_state.materiality_threshold} (Violation Rate):** `{n_violations}` (`{n_violations / total_pairs:.1%}`)")
                st.info(f"**Decision flipped (Decision Flip Rate):** `{n_flipped}` (`{n_flipped / total_pairs:.1%}`)")
                st.info(f"**Mean prediction delta:** `{st.session_state.violations_df['delta'].mean():.4f}`")
                st.info(f"**Max prediction delta:** `{st.session_state.violations_df['delta'].max():.4f}`")

                st.subheader("Worst Violations (Top 5 by delta):")
                worst_violations = st.session_state.violations_df.nlargest(5, 'delta')
                st.dataframe(worst_violations)

                st.markdown(f"""
                **Output Explanation and Real-World Impact:**

                The output provides concrete metrics for individual fairness. A **Violation Rate** of `{n_violations / total_pairs:.1%}` indicates that in `{n_violations / total_pairs:.1%}` of cases, similar applicants received significantly different credit approval probabilities. More critically, a **Decision Flip Rate** of `{n_flipped / total_pairs:.1%}` means `{n_flipped / total_pairs:.1%}` of twins, identical in all financial aspects, were given contradictory approve/deny decisions.

                As a Risk Manager, these numbers are direct evidence of potential individual discrimination. A high Decision Flip Rate, especially, is a major red flag, potentially leading to **adverse action notices** and **regulatory scrutiny under ECOA**. If Applicant A (with ZIP 10001) is approved and Applicant B (identical except ZIP 10456) is denied, Applicant B has grounds to request an explanation. If the explanation "your ZIP code contributed to the decline" is problematic due to its correlation with protected attributes, CreditGuard Financial faces significant liability. This quantification highlights the need for further investigation and potential model refinement.
                """)

                # V2: Violation Rate Histogram
                st.subheader("Visualization: Distribution of Prediction Deltas (V2)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(st.session_state.violations_df['delta'], bins=30, kde=True, color='skyblue', ax=ax)
                ax.axvline(x=st.session_state.materiality_threshold, color='red', linestyle='--', label=f'Materiality Threshold ($\epsilon={st.session_state.materiality_threshold:.2f}$)')
                ax.axvspan(st.session_state.materiality_threshold, st.session_state.violations_df['delta'].max(), color='red', alpha=0.1, label='Individual Fairness Violations')

                decision_flip_deltas = st.session_state.violations_df[st.session_state.violations_df['decision_flipped']]['delta']
                if not decision_flip_deltas.empty:
                    min_flip_delta = decision_flip_deltas.min()
                    ax.axvspan(min_flip_delta, st.session_state.violations_df['delta'].max(), color='purple', alpha=0.15, label='Decision Flip Region')

                ax.set_title('Distribution of Prediction Deltas Across Twin Pairs', fontsize=14)
                ax.set_xlabel('Absolute Prediction Probability Difference ($\Delta P$)', fontsize=12)
                ax.set_ylabel('Number of Twin Pairs', fontsize=12)
                ax.legend()
                ax.grid(axis='y', alpha=0.75)
                display_plot(fig)

                st.markdown(f"""
                The histogram visually reinforces the numerical findings. I can clearly see the distribution of prediction differences, how many fall above our materiality threshold ($\epsilon={st.session_state.materiality_threshold:.2f}$), and identify the region where decision flips occurred. This visualization is crucial for presenting the audit findings to non-technical stakeholders and regulatory bodies at CreditGuard Financial, making the concept of individual unfairness tangible.
                """)
            else:
                st.info("No violations detected yet. Click the 'Detect Violations' button.")

# --- Page 4: Decompose Unfairness (SHAP) ---
elif st.session_state.current_page == "4. Decompose Unfairness (SHAP)":
    st.header("4. Decomposing Unfairness with SHAP")
    st.markdown(f"""
    Identifying that a model produces unfair outcomes is one thing; understanding *why* is another. As a Risk Manager, after detecting individual fairness violations, my next step is to pinpoint the exact features driving these diverging predictions. For this, I will use **SHAP (SHapley Additive exPlanations)** values. SHAP helps decompose the prediction difference between an original applicant and their twin, attributing the impact to individual features.

    This process transforms the abstract finding of "the model is unfair" into "this specific feature causes unfairness for this specific applicant." This level of detail is invaluable for CreditGuard Financial to:
    1.  **Generate comprehensive Adverse Action Notices:** Explain to a denied applicant exactly why their profile led to that decision, even when a similar applicant was approved.
    2.  **Inform model developers:** Provide actionable insights on which features are disproportionately influencing outcomes for similar individuals.
    3.  **Bolster regulatory responses:** Demonstrate a deep understanding of model behavior and commitment to addressing bias at a granular level.
    """)

    if st.session_state.violations_df is None or st.session_state.twin_applicant_pairs is None:
        st.warning("Please run '2. Construct Twin Applicants' and '3. Detect Individual Violations' first.")
    else:
        st.subheader("Twin Applicant Gallery (Top 5 Worst Violations) (V1)")
        n_display_gallery = st.slider("Number of worst violations to display in gallery:", min_value=1, max_value=5, value=5)

        if st.button("Generate Twin Applicant Gallery"):
            st.markdown(f"<div style='border: 1px solid #ddd; padding: 15px; background-color: #f0f2f6;'><h3>Twin Applicant Gallery (Top {n_display_gallery} Worst Violations)</h3></div>", unsafe_allow_html=True)

            with st.spinner("Generating gallery and SHAP explanations... This may take a moment."):
                worst_violations = st.session_state.violations_df.nlargest(n_display_gallery, 'delta')

                for i, (_, row) in enumerate(worst_violations.iterrows()):
                    original_idx = row['original_idx']
                    pair = next(p for p in st.session_state.twin_applicant_pairs if p['original_idx'] == original_idx)

                    # Decompose difference for the pair
                    feature_impact, top_driver = decompose_twin_difference(
                        st.session_state.model, pair, st.session_state.feature_cols
                    )

                    st.markdown(f"""
                    <div style="border: 1px solid #eee; padding: 10px; margin-top: 15px; margin-bottom: 10px; background-color: #ffffff;">
                        <h4>Pair {i+1}: Original Index {original_idx} (Delta={row['delta']:.3f}) {("<<< DECISION FLIPPED >>>" if row['decision_flipped'] else "")}</h4>
                        <p><strong>Original Prediction: {row['orig_prob']:.3f}</strong> | <strong>Twin Prediction: {row['twin_prob']:.3f}</strong></p>
                        <p><strong>Primary Driver of Difference:</strong> {top_driver['feature']} (SHAP Diff: {top_driver['shap_diff']:.4f})</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Feature comparison table (adapted for Streamlit markdown coloring)
                    st.markdown("##### Features Comparison")
                    features_html = "<table><tr><th>Feature</th><th>Original Value</th><th>Twin Value</th><th>Type</th></tr>"
                    for col in st.session_state.feature_cols:
                        orig_val = pair['original'][col]
                        twin_val = pair['twin'][col]
                        type_str = ""
                        color = 'black'
                        if col in st.session_state.financial_features:
                            color = 'green'
                            type_str = "Financial (Identical)"
                        elif col in st.session_state.selected_proxy_features and orig_val != twin_val:
                            color = 'red'
                            type_str = "Proxy (Varied)"
                        else:
                            type_str = "Other (Identical)"
                        features_html += f"<tr><td>{col}</td><td style='color:{color};'>{orig_val:.2f}</td><td style='color:{color};'>{twin_val:.2f}</td><td>{type_str}</td></tr>"
                    features_html += "</table>"
                    st.markdown(features_html, unsafe_allow_html=True)


                    # Generate SHAP waterfall plots for this specific pair
                    explainer = shap.TreeExplainer(st.session_state.model)
                    orig_df_shap = pair['original'][st.session_state.feature_cols].to_frame().T
                    twin_df_shap = pair['twin'][st.session_state.feature_cols].to_frame().T

                    original_shap_values = explainer.shap_values(orig_df_shap)[0]
                    twin_shap_values = explainer.shap_values(twin_df_shap)[0]
                    expected_value = explainer.expected_value

                    fig_shap, axes_shap = plt.subplots(1, 2, figsize=(18, 6))

                    shap.waterfall_plot(shap.Explanation(values=original_shap_values, base_values=expected_value, data=orig_df_shap.iloc[0], feature_names=st.session_state.feature_cols), max_display=10, show=False, ax=axes_shap[0])
                    axes_shap[0].set_title(f'Original Applicant {original_idx} SHAP (P={row["orig_prob"]:.3f})')
                    axes_shap[0].set_xlabel('SHAP Value (impact on model output)')

                    shap.waterfall_plot(shap.Explanation(values=twin_shap_values, base_values=expected_value, data=twin_df_shap.iloc[0], feature_names=st.session_state.feature_cols), max_display=10, show=False, ax=axes_shap[1])
                    axes_shap[1].set_title(f'Twin Applicant SHAP (P={row["twin_prob"]:.3f})')
                    axes_shap[1].set_xlabel('SHAP Value (impact on model output)')

                    plt.tight_layout()
                    st.markdown("##### SHAP Waterfall Plots")
                    display_plot(fig_shap)

        st.markdown(f"""
        The "Twin Applicant Gallery" provides a compelling, visual narrative for each worst-case violation. For each pair, I can observe the identical financial features (highlighted in green) and the differing proxy features (highlighted in red). Crucially, the side-by-side SHAP waterfall plots for the worst cases illuminate *how* specific feature contributions, especially from the varied proxy features, lead to the divergent predictions.

        For example, if `ZIP_code_encoded` has a negative SHAP value for the original applicant (leading to approval) and a positive SHAP value for the twin (leading to denial), it unequivocally shows how that proxy feature drove the unfair outcome. This direct evidence is powerful for CreditGuard Financial to address regulatory concerns and improve model interpretability and fairness. The SHAP comparison for the single worst violation further emphasizes this by showing the exact shift in feature contributions.
        """)

        # V3: SHAP Waterfall Comparison for the single WORST violation pair
        st.subheader("SHAP Waterfall Comparisons for the SINGLE WORST VIOLATION PAIR (V3)")
        if st.session_state.violations_df is not None and not st.session_state.violations_df.empty:
            if st.button("Display Single Worst SHAP Comparison"):
                worst_pair_row = st.session_state.violations_df.nlargest(1, 'delta').iloc[0]
                worst_original_idx = worst_pair_row['original_idx']
                worst_pair_data = next(p for p in st.session_state.twin_applicant_pairs if p['original_idx'] == worst_original_idx)

                explainer = shap.TreeExplainer(st.session_state.model)
                worst_orig_df = worst_pair_data['original'][st.session_state.feature_cols].to_frame().T
                worst_twin_df = worst_pair_data['twin'][st.session_state.feature_cols].to_frame().T

                worst_original_shap_values = explainer.shap_values(worst_orig_df)[0]
                worst_twin_shap_values = explainer.shap_values(worst_twin_df)[0]
                expected_value = explainer.expected_value

                fig, axes = plt.subplots(1, 2, figsize=(20, 7))

                shap.waterfall_plot(shap.Explanation(values=worst_original_shap_values, base_values=expected_value, data=worst_orig_df.iloc[0], feature_names=st.session_state.feature_cols), max_display=10, show=False, ax=axes[0])
                axes[0].set_title(f'Worst Original Applicant {worst_original_idx} SHAP (P={worst_pair_row["orig_prob"]:.3f})', fontsize=14)
                axes[0].set_xlabel('SHAP Value (impact on model output)', fontsize=12)

                shap.waterfall_plot(shap.Explanation(values=worst_twin_shap_values, base_values=expected_value, data=worst_twin_df.iloc[0], feature_names=st.session_state.feature_cols), max_display=10, show=False, ax=axes[1])
                axes[1].set_title(f'Twin Applicant SHAP (P={worst_pair_row["twin_prob"]:.3f})', fontsize=14)
                axes[1].set_xlabel('SHAP Value (impact on model output)', fontsize=12)

                plt.tight_layout()
                plt.suptitle(f'SHAP Waterfall Comparison for Worst Violation Pair {worst_original_idx}', fontsize=16, y=1.02)
                display_plot(fig)
            else:
                st.info("Click the button above to display the SHAP Waterfall Comparison for the single worst violation.")

# --- Page 5: Measure Model Sensitivity (Lipschitz) ---
elif st.session_state.current_page == "5. Measure Model Sensitivity (Lipschitz)":
    st.header("5. Measuring Model Sensitivity (Lipschitz Fairness)")
    st.markdown(f"""
    Beyond specific twin pairs, I need to assess the model's overall robustness and sensitivity to minor perturbations in its input features. This is where **Lipschitz Fairness** comes in. The principle is that "similar individuals should receive similar predictions." A model is considered Lipschitz-fair if its prediction changes predictably (and not drastically) when input features change by a small amount.

    The practical metric we use is the **Proxy/Financial Sensitivity Ratio (R)**. This ratio compares the model's average prediction sensitivity to changes in proxy features versus changes in legitimate financial features.
    """)

    st.markdown(r"**Mathematical Formulation for Lipschitz Individual Fairness:**")
    st.markdown(r"A model $f$ satisfies $(\epsilon, \delta)$-individual fairness if:")
    st.markdown(r"$$d_y(f(x_1), f(x_2)) \le L \cdot d_x(x_1, x_2)$$")
    st.markdown(r"where $d_x(x_1, x_2)$ is a distance metric on input features (e.g., Euclidean distance); $d_y(f(x_1), f(x_2))$ is the prediction distance (e.g., absolute difference in probabilities); and $L$ is the Lipschitz constant, representing the maximum rate of change in output for a unit change in input.")
    st.markdown(f"Financial Interpretation: For two applicants with similar credit profiles ($d_x$ small on financial features), the model's predictions should also be similar ($d_y$ small). If the prediction changes dramatically when only a proxy feature changes, the Lipschitz constant with respect to proxy features is high, indicating individual unfairness.")

    st.markdown(r"The **Proxy/Financial Sensitivity Ratio (R)** is our practical metric:")
    st.markdown(r"$$R = \frac{{S_{\text{proxy}}}}{{S_{\text{financial}}}}$$")
    st.markdown(r"where $S_{\text{proxy}}$ is the average prediction sensitivity to small perturbations in proxy features; and $S_{\text{financial}}$ is the average prediction sensitivity to small perturbations in financial features.")

    st.markdown(f"**Interpretation:**")
    st.markdown(f"-   `R < 0.5`: Good. Model is much more sensitive to financial features than proxies.")
    st.markdown(f"-   `0.5 <= R < 1.0`: Warning. Proxy sensitivity is comparable.")
    st.markdown(f"-   `R >= 1.0`: Fail. Model is more sensitive to proxies than to legitimate credit factors.")

    st.markdown(f"""
    This analysis helps CreditGuard Financial understand if the model relies too heavily on proxy features even when minor variations occur, indicating a structural susceptibility to individual bias.
    """)

    if st.button("Measure Lipschitz Fairness"):
        with st.spinner("Calculating Lipschitz fairness and feature sensitivities..."):
            sensitivity_df, lipschitz_ratio_value = lipschitz_fairness(
                st.session_state.model_predict_proba,
                st.session_state.X_test,
                st.session_state.feature_cols,
                st.session_state.selected_proxy_features
            )
            st.session_state.sensitivity_df = sensitivity_df
            st.session_state.lipschitz_ratio_value = lipschitz_ratio_value
        st.success("Lipschitz fairness measurement complete!")

        if st.session_state.lipschitz_ratio_value is not None:
            st.subheader("Lipschitz Fairness Analysis Results")
            st.info(f"**Avg sensitivity to proxy features:** `{st.session_state.sensitivity_df[st.session_state.sensitivity_df['is_proxy']]['sensitivity'].mean():.4f}`")
            st.info(f"**Avg sensitivity to financial features:** `{st.session_state.sensitivity_df[~st.session_state.sensitivity_df['is_proxy']]['sensitivity'].mean():.4f}`")
            st.info(f"**Proxy/Financial sensitivity ratio (R):** `{st.session_state.lipschitz_ratio_value:.3f}`")

            if st.session_state.lipschitz_ratio_value < 0.5:
                st.success("PASS: Model is much more sensitive to financial features than proxies (R < 0.5).")
            elif st.session_state.lipschitz_ratio_value < 1.0:
                st.warning("WARNING: Proxy sensitivity is comparable to financial sensitivity (0.5 <= R < 1.0). Requires attention.")
            else:
                st.error("FAIL: Model is MORE sensitive to proxies than financials (R >= 1.0). Requires urgent intervention.")

            st.markdown(f"""
            **Output Explanation and Real-World Impact:**

            The Lipschitz Fairness Analysis provides a macro view of our model's fairness beyond specific twin pairs. The calculated Proxy/Financial Sensitivity Ratio (R) is a crucial indicator for CreditGuard Financial.
            -   If `R < 0.5`, it implies the model correctly prioritizes legitimate financial factors.
            -   If `0.5 <= R < 1.0`, it signals that proxy features have a surprisingly comparable influence, warranting further investigation.
            -   If `R >= 1.0`, it's a critical failure, indicating that the model is *more* sensitive to proxy features than to core financial attributes.

            This metric helps me, as a Risk Manager, assess the fundamental robustness of the model against subtle biases. A high ratio suggests that even small, seemingly innocuous changes in proxy features can dramatically swing credit decisions, posing a significant ethical and regulatory risk. This might necessitate a re-evaluation of model architecture or feature engineering to reduce undue influence from proxy features.
            """)

            # V4: Lipschitz Sensitivity Bar Chart
            st.subheader("Visualization: Lipschitz Sensitivity Bar Chart (V4)")
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(x='sensitivity', y='feature', data=st.session_state.sensitivity_df.sort_values('sensitivity', ascending=False),
                        palette=['red' if f in st.session_state.selected_proxy_features else 'blue' for f in st.session_state.sensitivity_df.sort_values('sensitivity', ascending=False)['feature']],
                        ax=ax)
            ax.set_title('Feature Sensitivity to Prediction Changes (Lipschitz Analysis)', fontsize=14)
            ax.set_xlabel('Average Prediction Sensitivity (per unit feature change)', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.grid(axis='x', alpha=0.75)
            ax.legend(handles=[plt.Line2D([0], [0], color='red', lw=4, label='Proxy Feature'),
                               plt.Line2D([0], [0], color='blue', lw=4, label='Financial Feature')],
                      title='Feature Type')
            display_plot(fig)

            st.markdown(f"""
            The Lipschitz Sensitivity Bar Chart clearly ranks features by their influence on prediction changes. By color-coding proxy features (red) and financial features (blue), I can visually identify if proxy features exhibit unexpectedly high sensitivity. This visualization makes it easy to communicate complex fairness insights to CreditGuard Financial's executive team and model developers, highlighting which features, if perturbed slightly, would cause the largest shifts in credit decisions.
            """)
        else:
            st.info("No Lipschitz fairness results yet. Click the 'Measure Lipschitz Fairness' button.")

# --- Page 6: Adversarial Debiasing (Conceptual) ---
elif st.session_state.current_page == "6. Adversarial Debiasing (Conceptual)":
    st.header("6. Conceptual Understanding of Adversarial Debiasing (Mitigation Strategy)")
    st.markdown(f"""
    While our audit focuses on *detecting* and *verifying* individual fairness, it's essential for me as a Risk Manager to also understand *mitigation* strategies. **Adversarial Debiasing** is a powerful, albeit complex, technique for mitigating bias directly during model training. It aims to build a model whose internal representations do not encode information about protected attributes, even indirectly through proxy features.

    The core idea involves training two competing neural networks:
    1.  **Predictor (↑):** A main model that predicts the target variable (e.g., credit default).
    2.  **Adversary (↓):** A separate model that tries to predict the protected attribute (e.g., race, gender, or proxy for them like `ZIP_code`) from the *intermediate representations* learned by the predictor.

    The predictor is trained to minimize its prediction error *while simultaneously* trying to "fool" the adversary. This is achieved using a **gradient reversal layer** that inverts the gradients from the adversary, effectively forcing the predictor's encoder to learn representations that are *independent* of the protected attribute. If the adversary cannot predict the protected attribute from the predictor's internal representations, then those representations (and thus the final predictions) are considered 'fair' with respect to that attribute.
    """)

    st.markdown(r"**Mathematical Concept: Gradient Reversal Layer**")
    st.markdown(r"The gradient reversal layer (GRL) is a crucial component in adversarial debiasing. During the forward pass, it acts as an identity function, simply passing its input unchanged:")
    st.markdown(r"$$f_{\text{GRL}}(x) = x$$")
    st.markdown(r"However, during the backward pass (gradient calculation), it multiplies the gradient by a negative constant $\lambda$:")
    st.markdown(r"$$\frac{{\partial L}}{{\partial x}} = -\lambda \frac{{\partial L}}{{\partial f_{\text{GRL}}(x)}}$$")
    st.markdown(r"This effectively reverses the direction of the gradient flow for the adversary's loss, making the feature extractor (encoder) learn features that confuse the adversary while still being useful for the main predictor.")

    st.markdown(f"""
    This conceptual understanding is vital for CreditGuard Financial to consider advanced bias mitigation techniques for future model development, especially when high individual fairness is a critical requirement.
    """)

    st.subheader("Adversarial Debiasing Architecture Diagram (V6)")
    st.markdown(f"""
    <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 20px;">
        <p style="font-size: 1.1em; font-weight: bold;">Conceptual Adversarial Debiasing Architecture</p>
        <p>This diagram illustrates the conceptual architecture for adversarial debiasing. The goal is for the 'Encoder' to learn representations that are useful for the 'Predictor' BUT simultaneously 'fool' the 'Adversary' into not being able to predict the protected attribute.</p>
        <pre><code>
Encoder (Feature Extractor) -> ( Predictor (Credit Default) ↑ , Adversary (Protected Attribute) ↓ )
                          ^                                                        |
                          |----- Gradient Reversal Layer ---------------------------|
                          |
                     Input Features
        </code></pre>
        <p>Adversarial debiasing trains two competing objectives:</p>
        <ul>
            <li>Predictor: minimize credit default prediction error</li>
            <li>Adversary: predict protected attribute from representation</li>
            <li>Encoder: fool the adversary while keeping predictor accurate</li>
        </ul>
        <p>If the adversary CANNOT predict group membership from the learned representation, the model cannot discriminate with respect to that attribute.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"Conceptual `AdversarialDebiaser` model instantiated with `{len(st.session_state.feature_cols)}` features and `64` hidden dimension.")
    st.markdown(f"Model Architecture:")
    # Using st.code to display the architecture string from the class.
    st.code(str(AdversarialDebiaser(len(st.session_state.feature_cols))))

    st.markdown(f"""
    **Output Explanation and Real-World Impact:**

    The conceptual architecture diagram and the `AdversarialDebiaser` class visually and programmatically illustrate how this advanced technique works. As a Risk Manager, understanding adversarial debiasing's mechanism, particularly the role of the gradient reversal layer, allows me to appreciate its potential to build "fair by design" models.

    This method forces the model to learn intermediate representations that are disentangled from protected attributes, fundamentally reducing the risk of indirect bias. However, it's a trade-off: aggressively removing all protected attribute information might sometimes reduce predictive accuracy. For CreditGuard Financial, this technique is most appropriate when regulatory risk is severe, and the highest standards of individual fairness are paramount, even at a slight cost to overall model performance. It informs future strategic decisions on model development and ethical AI governance.
    """)

# --- Page 7: Synthesize Fairness Audit Findings ---
elif st.session_state.current_page == "7. Synthesize Fairness Audit Findings":
    st.header("7. Synthesizing Fairness Audit Findings")
    st.markdown(f"""
    The final step in my role as a Risk Manager is to compile all findings from our comprehensive fairness audit into a concise synthesis report. This report is crucial for providing a holistic view of the model's fairness posture across detection, mitigation (conceptualized here), and individual verification phases. It summarizes key metrics and an overall assessment for CreditGuard Financial's stakeholders, including compliance, legal, and executive leadership.

    This synthesis report helps CreditGuard Financial to:
    -   **Consolidate evidence:** Present a unified picture of fairness performance.
    -   **Inform strategy:** Guide decisions on model adjustments, policy changes, or further mitigation efforts.
    -   **Demonstrate commitment:** Provide clear documentation of our rigorous fairness testing process.

    To provide a complete picture for the synthesis report, we will use placeholder values for `group_metrics` and `mitigation_results` from prior (conceptual) group fairness analyses (D4-T2-C1 and D4-T2-C2). This demonstrates the comprehensive nature of the full fairness lifecycle.
    """)

    if st.session_state.violations_df is None or st.session_state.lipschitz_ratio_value is None:
        st.warning("Please complete '3. Detect Individual Violations' and '5. Measure Model Sensitivity (Lipschitz)' first.")
    else:
        if st.button("Generate Synthesis Report"):
            with st.spinner("Compiling comprehensive fairness audit report..."):
                synthesis_report = topic2_synthesis_report(
                    st.session_state.conceptual_group_metrics,
                    st.session_state.conceptual_mitigation_results,
                    st.session_state.violations_df,
                    st.session_state.lipschitz_ratio_value
                )
                st.session_state.synthesis_report = synthesis_report
            st.success("Synthesis report generated!")

            if st.session_state.synthesis_report is not None:
                st.subheader("Comprehensive AI Fairness Assessment")
                
                for key, value in st.session_state.synthesis_report.items():
                    if isinstance(value, dict):
                        st.markdown(f"#### {key.replace('_', ' ').title()}")
                        for sub_key, sub_value in value.items():
                            st.markdown(f"  **{sub_key.replace('_', ' ').title()}:** `{sub_value}`")
                    elif isinstance(value, list):
                        st.markdown(f"#### {key.replace('_', ' ').title()}")
                        for item in value:
                            st.markdown(f"- `{item}`")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** `{value}`")

                st.markdown(f"""
                **Output Explanation and Real-World Impact:**

                The `COMPREHENSIVE AI FAIRNESS ASSESSMENT` provides a dashboard for CreditGuard Financial to quickly grasp the model's fairness profile. It integrates findings from group-level analysis (conceptualized as D4-T2-C1 and D4-T2-C2) with the individual-level audit conducted in this notebook.

                The report clearly states the `Decision Flip Rate` and `Lipschitz Ratio`, along with an `individual_fairness_assessment`. If the `overall_assessment` is "REQUIRES FURTHER MITIGATION", it immediately signals to the executive team that despite passing group-level tests, the model exhibits concerning individual-level biases. The `remaining_actions` section then provides concrete, actionable steps for CreditGuard Financial's compliance and development teams. This synthesis is the ultimate deliverable, guiding strategic decisions on responsible AI and demonstrating our commitment to fair dealing in accordance with CFA Standard III(B).
                """)

                # V5: Topic 2 Synthesis Dashboard
                st.subheader("Visualization: Topic 2 Synthesis Dashboard (V5)")
                fig = plt.figure(figsize=(14, 8))
                gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])

                # Phase 1: Group Detection Summary
                ax0 = plt.subplot(gs[0, 0])
                ax0.text(0.5, 0.7, f"Phase 1: Group Detection (D4-T2-C1)", ha='center', va='center', fontsize=14, weight='bold')
                ax0.text(0.5, 0.4, f"DIR: {st.session_state.conceptual_group_metrics['dir']:.3f} ({'PASS' if st.session_state.conceptual_group_metrics['four_fifths_rule_pass'] else 'FAIL'})", ha='center', va='center', fontsize=12)
                ax0.text(0.5, 0.1, f"Proxies Detected: {st.session_state.conceptual_group_metrics['proxies_detected']}", ha='center', va='center', fontsize=12)
                ax0.set_title('Group Fairness Detection', fontsize=16, pad=15)
                ax0.axis('off')

                # Phase 2: Group Mitigation Summary
                ax1 = plt.subplot(gs[0, 1])
                ax1.text(0.5, 0.7, f"Phase 2: Group Mitigation (D4-T2-C2)", ha='center', va='center', fontsize=14, weight='bold')
                ax1.text(0.5, 0.4, f"Strategy: {st.session_state.conceptual_mitigation_results['strategy_applied']}", ha='center', va='center', fontsize=12)
                ax1.text(0.5, 0.1, f"Post-Mitigation DIR: {st.session_state.conceptual_mitigation_results['post_mitigation_dir']:.3f} | AUC Cost: {st.session_state.conceptual_mitigation_results['auc_cost']:.2%}", ha='center', va='center', fontsize=12)
                ax1.set_title('Group Fairness Mitigation (Conceptual)', fontsize=16, pad=15)
                ax1.axis('off')

                # Phase 3: Individual Verification Summary
                ax2 = plt.subplot(gs[1, :])
                ax2.text(0.5, 0.85, f"Phase 3: Individual Verification (D4-T2-C3)", ha='center', va='center', fontsize=14, weight='bold')
                ax2.text(0.2, 0.6, f"Twin Pairs Tested: {len(st.session_state.violations_df)}", ha='center', va='center', fontsize=12)
                ax2.text(0.8, 0.6, f"Decision Flips: {st.session_state.violations_df['decision_flipped'].sum()} ({st.session_state.violations_df['decision_flipped'].mean():.1%})", ha='center', va='center', fontsize=12)
                ax2.text(0.2, 0.35, f"Materiality Violations: {st.session_state.violations_df['is_violation'].sum()} ({st.session_state.violations_df['is_violation'].mean():.1%})", ha='center', va='center', fontsize=12)
                ax2.text(0.8, 0.35, f"Lipschitz Ratio (R): {st.session_state.lipschitz_ratio_value:.3f}", ha='center', va='center', fontsize=12)
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
                display_plot(fig)
            else:
                st.info("No synthesis report generated yet. Click the 'Generate Synthesis Report' button.")
```

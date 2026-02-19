#!/usr/bin/env python3
"""
Test script to verify that all imports and variables are correctly set up
between app.py and source.py
"""

print("Testing imports from source.py...")
print("=" * 60)

try:
    from source import (
        model, X_test, y_test, X_train, y_train,
        feature_cols, financial_features, proxy_features,
        model_predict_proba, conceptual_group_metrics,
        conceptual_mitigation_results,
        construct_twin_pairs, detect_individual_violations,
        decompose_twin_difference, lipschitz_fairness,
        topic2_synthesis_report, AdversarialDebiaser,
        plot_shap_waterfall_comparison
    )

    print("✓ All imports successful!")
    print()
    print(f"✓ model type: {type(model)}")
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ y_test shape: {y_test.shape}")
    print(f"✓ feature_cols: {len(feature_cols)} features")
    print(f"✓ financial_features: {financial_features}")
    print(f"✓ proxy_features: {proxy_features}")
    print(f"✓ conceptual_group_metrics: {conceptual_group_metrics}")
    print(
        f"✓ conceptual_mitigation_results keys: {list(conceptual_mitigation_results.keys())}")
    print()
    print(f"✓ construct_twin_pairs: {callable(construct_twin_pairs)}")
    print(
        f"✓ detect_individual_violations: {callable(detect_individual_violations)}")
    print(
        f"✓ decompose_twin_difference: {callable(decompose_twin_difference)}")
    print(f"✓ lipschitz_fairness: {callable(lipschitz_fairness)}")
    print(f"✓ topic2_synthesis_report: {callable(topic2_synthesis_report)}")
    print(f"✓ AdversarialDebiaser: {AdversarialDebiaser}")
    print()
    print("=" * 60)
    print("All imports and variables are correctly configured!")
    print("=" * 60)

except Exception as e:
    print(f"✗ Import failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

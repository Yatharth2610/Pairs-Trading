"""
Explainability module using SHAP.
"""
import shap
import matplotlib.pyplot as plt

def generate_shap_explanations(clf, X_test, feature_names):
    """
    Generate SHAP plots for the Random Forest model.
    Since we use CalibratedClassifierCV, we extract the base estimator.
    """
    try:
        # Extract the Random Forest from the CalibratedClassifierCV
        # Usually it creates an ensemble of calibrated classifiers, we take the first one's base estimator
        trained_rf = clf.calibrated_classifiers_[0].estimator
    except AttributeError:
        # If it wasn't calibrated or has different structure, try using directly
        trained_rf = clf
        
    assert hasattr(trained_rf, "estimators_"), "Model does not seem to be a fitted Random Forest"
    
    explainer = shap.TreeExplainer(trained_rf)
    
    # SHAP values for the positive class (class 1)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        # Depending on SHAP version, it might return a 3D array (samples, features, classes)
        if len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values
            
    print("Generating SHAP Bar Plot (Global Feature Importance)...")
    shap.summary_plot(
        shap_values_class1,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.show()
    
    print("Generating SHAP Beeswarm Plot (Feature Impacts)...")
    shap.summary_plot(
        shap_values_class1,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.show()
    
    return explainer, shap_values_class1

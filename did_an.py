import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_did_intervention_analysis(ward_name, pre_hours=12, post_hours=12):
    """
    Simulates / ingests panel data across treated ward vs control wards
    to estimate average treatment effect (ATT) via OLS:
    AQI ~ Treated + Post + (Treated * Post)
    """
    np.random.seed(42)
    records = []

    # Pre & post periods
    periods = [f"t-{i}" for i in range(pre_hours, 0, -1)] + [f"t+{i}" for i in range(1, post_hours + 1)]

    # Treated Ward (intervened)
    for idx, t in enumerate(periods):
        is_post = 1 if idx >= pre_hours else 0
        # Treatment effect reduces AQI by ~32 points post-intervention
        aqi = 210 + np.random.normal(0, 4) - (32 * is_post)
        records.append({
            "Ward": ward_name,
            "Treated": 1,
            "Post": is_post,
            "AQI": aqi,
            "Period": t
        })

    # Control Ward (Nerul / residential baseline without misting deployment)
    for idx, t in enumerate(periods):
        is_post = 1 if idx >= pre_hours else 0
        # Natural variation without targeted intervention
        aqi = 160 + np.random.normal(0, 4) + (2 * is_post)
        records.append({
            "Ward": "Nerul (Control)",
            "Treated": 0,
            "Post": is_post,
            "AQI": aqi,
            "Period": t
        })

    df_panel = pd.DataFrame(records)

    # Standard 2x2 DiD Fixed Effects / OLS Regression
    model = smf.ols("AQI ~ Treated * Post", data=df_panel).fit()
    treatment_effect = model.params["Treated:Post"]
    p_value = model.pvalues["Treated:Post"]

    pre_treatment_avg = df_panel[(df_panel["Treated"] == 1) & (df_panel["Post"] == 0)]["AQI"].mean()
    post_treatment_avg = df_panel[(df_panel["Treated"] == 1) & (df_panel["Post"] == 1)]["AQI"].mean()

    return {
        "treatment_effect": round(treatment_effect, 2),
        "p_value": round(p_value, 4),
        "pre_avg": round(pre_treatment_avg, 1),
        "post_avg": round(post_treatment_avg, 1),
        "df_panel": df_panel
    }

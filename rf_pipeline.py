import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

def prepare_data(df, disease, seed, ts_mode):
    # Predictor selection
    cols_org = [x for x in df.columns if x not in [
        "FAKT_FID", 
        "Outcome_HWR_indexes", "Outcome_HWR_readmission_is",
        "Outcome_Surg_indexes", "Outcome_Surg_readmission_is", 
        "Outcome_Med_all_indexes", "Outcome_Med_all_readmission_is"
    ]]

    cols = [x for x in cols_org if x in df.columns]
    mask = df[f"Outcome_{disease}_indexes"] == 1
    X = df.loc[mask, cols].copy()
    y = df.loc[mask, f"Outcome_{disease}_readmission_is"]
    ids = df.loc[mask, "FAKT_FID"]
    
    # Train-test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=seed, stratify=y
    )

    # Imputation
    for col in cols:
        X_train[col].fillna(X_train[col].dropna().median(), inplace=True)
        X_test[col].fillna(X_test[col].dropna().median(), inplace=True)
        
    # Feature selection
    feature_groups = [
        "Blutdruck_sys", "Blutdruck_dia", "SPO2", "Herzfrequenz", 
        "Temperatur", "Schmerz_Bewegung", "Schmerz_Ruhe", "POCT",
        "FAKT_Volume_Treating_Physician_Cases_Prev_", "Elix_CI_HOSP_Assign", 
        "FAKT_ED_Encounter_", "FAKT_HospitalStay_", "FAKT_OPS_Counter",
        "FAKT_OutpatientVisits_", "FAKT_Cases_", "HospitalStay_",
        "Scores_DOS", "Scores_GCS", "Scores_NRS", "Scores_SPI",
        "EPA_Char_Dekubitusrisiko (Braden)", "EPA_Char_Pneumonierisiko", 
        "EPA_Char_Risiko poststationäres Versorgungsdefizit", "EPA_Char_Sturzrisiko"
    ]

    for group in feature_groups:
        vars_group = [x for x in X_train.columns if x.startswith(group)]
        if not vars_group:
            continue
        select = SelectKBest(f_classif, k=1)
        try:
            select.fit(X_train[vars_group], y_train)
            selected = select.get_feature_names_out()
            vars_to_drop = [x for x in vars_group if x not in selected]
            X_train.drop(columns=vars_to_drop, inplace=True)
            X_test.drop(columns=vars_to_drop, inplace=True)
        except Exception:
            continue
    
    # Optional: Drop low-importance features in second run
    if ts_mode == 2:
        drop_fraction = 0.25
        importances = pd.Series(np.random.rand(len(X_train.columns)), index=X_train.columns)
        threshold = importances.quantile(drop_fraction)
        to_drop = importances[importances <= threshold].index
        X_train.drop(columns=to_drop, inplace=True)
        X_test.drop(columns=to_drop, inplace=True)

    return X_train, X_test, y_train, y_test, ids_test

def train_evaluate(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [500],
        'max_depth': [15],
        'max_features': ['sqrt'],
        'min_samples_split': [10],
        'min_samples_leaf': [5],
        'criterion': ['gini'],
        'class_weight': [None]
    }

    grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_

    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC Test: {auc_score:.3f}")

    return best_rf, y_pred_prob, auc_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_data.csv", help="CSV input file")
    parser.add_argument("--disease", choices=["HWR", "Surg", "Med_all"], default="Med_all")
    parser.add_argument("--ts_mode", type=int, choices=[1, 2], default=1, help="Feature selection mode")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    for seed in range(1, 21):  # 20 train-test splits
        print(f"\n### Train-Test Split {seed}/20 ###")
        X_train, X_test, y_train, y_test, ids_test = prepare_data(df, args.disease, seed, args.ts_mode)
        model, y_pred_prob, auc_score = train_evaluate(X_train, X_test, y_train, y_test)

        # Save feature importances
        importances = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        importances.to_csv(f"feature_importances_{args.disease}_{seed}.csv", index=False)

        # Save predicted probabilities
        predictions = pd.DataFrame({
            "FAKT_FID": ids_test,
            "y_true": y_test,
            "y_pred_prob": y_pred_prob
        })
        predictions.to_csv(f"predicted_probs_{args.disease}_{seed}.csv", index=False)

        print("Saved predictions and importances.")
        gc.collect()

if __name__ == "__main__":
    main()

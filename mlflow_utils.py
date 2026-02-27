import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterSampler, cross_validate


def run_tuning_mlflow(
    experiment_name: str, 
    run_name: str, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model, 
    param_dist: dict, 
    n_iter: int = 10, 
    cv: int = 5, 
    username: str = 'Abdelaziz Ahmed', 
    description: str = None
):
    """
    Performs RandomizedSearchCV manually to log custom run names, input examples, 
    merged datasets, and multiple evaluation metrics (Accuracy, Precision, Recall, F1).
    """
    mlflow.set_experiment(experiment_name)
    
    # Disable autolog for granular control over the tracking
    mlflow.sklearn.autolog(disable=True)
    
    print(f"Starting Hyperparameter Tuning: {run_name}")

    # Generate parameter list
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    
    # Data prep for logging input examples safely
    input_example = X_train.iloc[:5].to_numpy() if hasattr(X_train, 'iloc') else X_train[:5]

    with mlflow.start_run(run_name=run_name) as parent_run:
        # 1. Log Parent Tags
        tags = {
            "mlflow.user": username,
            "model_type": type(model).__name__,
            "tuning_run": "parent"
        }
        if description:
            tags["description"] = description
        mlflow.set_tags(tags)

        # 2. Merge X_train and y_train, then log as an MLflow Dataset
        try:
            combined_df = pd.concat([X_train, y_train], axis=1)
            target_col = y_train.name if y_train.name else combined_df.columns[-1]
            
            dataset = mlflow.data.from_pandas(
                combined_df, 
                targets=target_col, 
                name=f"{run_name}_training_data"
            )
            
            mlflow.log_input(dataset, context="training")
            print("Merged dataset logged successfully to MLflow.")
            
        except Exception as e:
            print(f"Warning: Failed to merge and log dataset. Error: {e}")

        # 3. Hyperparameter Tuning Loop
        best_score = -1
        best_params = None
        best_metrics = {}
        results = []
        
        # Define the multiple metrics we want to evaluate
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        for i, params in enumerate(param_list):
            trial_name = f"{run_name}_Trial_{i+1}"
            print(f"Running {trial_name} with params: {params}")
            
            with mlflow.start_run(run_name=trial_name, nested=True):
                mlflow.set_tags({
                    "mlflow.user": username,
                    "parent_run_id": parent_run.info.run_id
                })
                mlflow.log_params(params)
                
                # Set params and cross-validate with multiple metrics
                model.set_params(**params)
                cv_results = cross_validate(
                    model, X_train, y_train, 
                    cv=cv, 
                    scoring=scoring_metrics, 
                    return_train_score=False
                )
                
                # Extract mean scores for the current trial
                mean_accuracy = np.mean(cv_results['test_accuracy'])
                mean_precision = np.mean(cv_results['test_precision_macro'])
                mean_recall = np.mean(cv_results['test_recall_macro'])
                mean_f1 = np.mean(cv_results['test_f1_macro'])
                
                # Log all metrics for this trial
                trial_metrics = {
                    "mean_cv_accuracy": mean_accuracy,
                    "mean_cv_precision_macro": mean_precision,
                    "mean_cv_recall_macro": mean_recall,
                    "mean_cv_f1_macro": mean_f1
                }
                mlflow.log_metrics(trial_metrics)
                
                # Track best using F1-macro as the primary decision metric
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_params = params
                    best_metrics = {f"best_{k}": v for k, v in trial_metrics.items()}
                
                # Fit and Log Model Artifact
                model.fit(X_train, y_train)
                prediction = model.predict(input_example)
                signature = infer_signature(input_example, prediction)
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
                

        # 4. Log Best Results to Parent
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(best_metrics)
        
        print(f"Tuning Complete. Best F1-Macro Score: {best_score:.4f}")
        print(f"Best Params: {best_params}")


def load_registered_model(model_name: str):
    """
    Loads the latest version of a model from the registry.
    Returns the loaded model or None if failed.
    """
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name)
        if not versions:
            print(f"No versions found for model '{model_name}'.")
            return None
            
        latest_version = max(versions, key=lambda v: int(v.version))
        print(f"Loading version {latest_version.version} (Stage: {latest_version.current_stage})")
        
        model_uri = f"models:/{model_name}/{latest_version.version}"
        return mlflow.sklearn.load_model(model_uri)
        
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None
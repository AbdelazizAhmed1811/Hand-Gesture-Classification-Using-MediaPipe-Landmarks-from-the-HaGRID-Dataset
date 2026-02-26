import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterSampler, cross_validate
from sklearn.base import clone

from mlflow.tracking import MlflowClient



def run_mlflow(experiment_name, run_name, X_train, y_train, model, X_test=None, y_test=None, username='Abdelaziz Ahmed', description=None):
    """
    Runs MLflow for training a model using mlflow.evaluate.
    """
    
    mlflow.set_experiment(experiment_name)
    

    mlflow.sklearn.autolog(disable=True) 

    with mlflow.start_run(run_name=run_name) as run:
        
        # Set Tags
        mlflow.set_tag("mlflow.user", username)
        mlflow.set_tag("model_type", type(model).__name__)
        if description:
            mlflow.set_tag("description", description)
        
        # Train Model
        model.fit(X_train, y_train)
        

        if hasattr(X_train, 'iloc'):
            input_example = X_train.iloc[:5]
        else:
            input_example = X_train[:5]
            
        prediction = model.predict(input_example)
        signature = infer_signature(input_example, prediction)
        
        # Log Model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        # Evaluate if Test Data is provided
        if X_test is not None and y_test is not None:
            print("Evaluating on Test Data...")
            
            eval_result = mlflow.evaluate(
                model=model_info.model_uri,
                data=X_test,
                targets=y_test,
                model_type="classifier",
                evaluator_config={"log_confusion_matrix": True}
            )
            
            print(f"Evaluation Metrics: {eval_result.metrics}")
            
    print(f"Run Complete: {run_name}")


def run_cross_validation_mlflow(experiment_name, run_name, X, y, model, cv=5, username='Abdelaziz Ahmed', description=None):
    """
    Runs Cross-Validation and logs results using mlflow.evaluate.
    """
    
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(disable=True)
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)


    is_df = hasattr(X, 'iloc')
    
    print(f"Starting Cross-Validation: {run_name}")

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tag("mlflow.user", username)
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("cv_run", "parent")
        if description:
            mlflow.set_tag("description", description)

        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        
        fold = 1

        # To handle both consistently:
        if is_df:
            X_data = X.values
            y_data = y.values
        else:
            X_data = X
            y_data = y

        for train_idx, val_idx in kf.split(X_data):
            fold_run_name = f"{run_name}_Fold_{fold}"
            
            with mlflow.start_run(run_name=fold_run_name, nested=True) as child_run:
                mlflow.set_tag("fold", fold)
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)
                mlflow.set_tag("mlflow.user", username)
                mlflow.set_tag("model_type", type(model).__name__)
                
                # Split Data
                X_t, X_v = X_data[train_idx], X_data[val_idx]
                y_t, y_v = y_data[train_idx], y_data[val_idx]
                
                # Clone and Train
                fold_model = clone(model)
                fold_model.fit(X_t, y_t)
                
                # Input Example & Signature
                input_example = X_t[:5]
                prediction = fold_model.predict(input_example)
                signature = infer_signature(input_example, prediction)
                
                # Log Model
                model_info = mlflow.sklearn.log_model(
                    sk_model=fold_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
                

                eval_result = mlflow.evaluate(
                    model=model_info.model_uri,
                    data=X_v,
                    targets=y_v,
                    model_type="classifier"
                )
                

                metrics = eval_result.metrics
                

                
                acc = metrics.get('accuracy', 0)

                fold_metrics['accuracy'].append(acc)

                print(f"Fold {fold} - Accuracy: {acc:.4f}")
                
            fold += 1
        
        # Log Aggregate Metrics to Parent Run
        mean_acc = np.mean(fold_metrics['accuracy'])
        std_acc = np.std(fold_metrics['accuracy'])
        
        mlflow.log_metric("mean_cv_accuracy", mean_acc)
        mlflow.log_metric("std_cv_accuracy", std_acc)
        
        print(f"CV Complete. Mean Accuracy: {mean_acc:.4f} \u00b1 {std_acc:.4f}")


def run_tuning_mlflow(experiment_name, run_name, X_train, y_train, model, param_dist, n_iter=10, cv=5, username='Abdelaziz Ahmed', description=None):
    """
    Performs RandomizedSearch CV manually to log custom run names and input examples.
    """
    mlflow.set_experiment(experiment_name)
    
    # Disable autolog for granular control
    mlflow.sklearn.autolog(disable=True)
    
    print(f"Starting Hyperparameter Tuning: {run_name}")

    # Generate parameter list
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    
    # Data prep for logging
    if hasattr(X_train, 'iloc'):
        input_example = X_train.iloc[:5]
    else:
        input_example = X_train[:5]

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tag("mlflow.user", username)
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("tuning_run", "parent")
        if description:
            mlflow.set_tag("description", description)

        best_score = -1
        best_params = None
        results = []

        for i, params in enumerate(param_list):
            trial_name = f"{run_name}_Trial_{i+1}"
            print(f"Running {trial_name} with params: {params}")
            
            with mlflow.start_run(run_name=trial_name, nested=True) as child_run:
                mlflow.set_tag("mlflow.user", username)
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)
                mlflow.log_params(params)
                
                # Set params
                model.set_params(**params)
                
                # Cross Validate
                cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring='accuracy', return_train_score=False)
                mean_score = np.mean(cv_results['test_score'])
                
                mlflow.log_metric("mean_cv_accuracy", mean_score)
                
                # Track best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                
                # Fit and Log Model Artifact (Requirement: input sample for ALL models)
                model.fit(X_train, y_train)
                prediction = model.predict(input_example)
                signature = infer_signature(input_example, prediction)
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
                
                # Collect results
                params['mean_test_score'] = mean_score
                results.append(params)

        # Log Best Results to Parent
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy", best_score)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv("tuning_results.csv", index=False)
        mlflow.log_artifact("tuning_results.csv")
        
        print(f"Tuning Complete. Best Score: {best_score:.4f}")
        print(f"Best Params: {best_params}")


def load_registered_model(model_name):
    """
    Loads the latest version of a model from the registry.
    Returns the loaded model or None if failed.
    """
    client = MlflowClient()
    try:
        # Get the latest versions (returns a list of ModelVersion objects)
        versions = client.get_latest_versions(model_name)
        if not versions:
            print(f"No versions found for model '{model_name}'.")
            return None
        latest_version = max(versions, key=lambda v: int(v.version))
        print(f"Loading version {latest_version.version} (Stage: {latest_version.current_stage})")
        
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

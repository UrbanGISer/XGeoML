import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from .preprocess import *


#04 keep trained models
class train_model:
    def _train_point_sklearn_model(df, feature_names, target_name, point_index, weights, model_class):
        X_train, y_train, X_test = preprocess. _preprocess_data(df, feature_names, target_name, point_index, weights)
        model = model_class()  # Assuming model_class is defined elsewhere
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        return y_pred,model

    def train_sklearn(df, feature_names, target_name, weights, model_class):
        trained_models = {}
        predictions = np.zeros(len(df))
        results = Parallel(n_jobs=-1)(
            delayed(train_model._train_point_sklearn_model)(df, feature_names, target_name, i, weights, model_class) for i in tqdm(range(len(df)), desc="Computing Prediction")
        )
        for i, (y_pred, model) in enumerate(results):
            predictions[i] = y_pred
            trained_models[i] = model
        return trained_models,predictions

    def train_hysklearn(df, feature_names, target_name, weights, hymodel,max_evals=10, trial_timeout=60):
        from hpsklearn import HyperoptEstimator
        from hyperopt import tpe
        trained_models = {}
        predictions = np.zeros(len(df))
        for point_index in tqdm(range(len(df)), desc="Train Model"):
            X_train, y_train, X_test = preprocess. _preprocess_data(df, feature_names, target_name, point_index, weights)
            for attempt in range(0, 100):
                try:
                    tmodel = HyperoptEstimator(regressor=hymodel("myModel"), preprocessing=[], algo=tpe.suggest, max_evals=max_evals, trial_timeout=trial_timeout, seed=1)
                    tmodel.fit(X_train, y_train)
                    y_pred = tmodel.predict(X_test)[0]
                    trained_models[point_index] = tmodel
                    predictions[point_index] = y_pred
                    break  
                except Exception as e:
                    print(f"Attempt {attempt+1}: Model training failed with error: {e}")
        return trained_models,predictions

    def _explain_point_model(df, feature_names, target_name, point_index, weights, trained_models, explainer_names,skleanrmodel=True):
        X_train, y_train, X_test = preprocess. _preprocess_data(df, feature_names, target_name, point_index, weights)
        model = trained_models[point_index]
        importances = {}
        if 'LIME' in explainer_names:
            importances['LIME'] = preprocess._calculate_lime_importances(model, X_train, X_test, feature_names, target_name)
        if skleanrmodel==True:
            if 'SHAP' in explainer_names:
                importances['SHAP'] = preprocess._calculate_shap_importances(model, X_train, X_test) 
            if 'Importance' in explainer_names and hasattr(model, 'feature_importances_'):
                importances['Importance'] = model.feature_importances_   
        else:
            if 'SHAP' in explainer_names:
                importances['SHAP'] = preprocess._calculate_shap_importances(model.predict, X_train, X_test) 
            if 'Importance' in explainer_names and hasattr(model.best_model()['learner'], 'feature_importances_'):
                importances['Importance'] = model.best_model()['learner'].feature_importances_  
        importance_combine = np.concatenate([value.reshape(-1) for value in importances.values()])
        return importance_combine

    def explain_models(df, feature_names, target_name, weights, trained_models, explainer_names,skleanrmodel=True):
        #pretest
        explainerlist= explainer_names.copy()   
        X_train1, y_train1, _ =preprocess._preprocess_data(df, feature_names, target_name, 1, weights)
        model = trained_models[1] # Assuming model_class is defined elsewhere
        model.fit(X_train1, y_train1)
        if skleanrmodel==True:
            model_supports_importance = hasattr(trained_models[0], 'feature_importances_')            
        else:
            model_supports_importance = hasattr(trained_models[0].best_model()['learner'], 'feature_importances_')
        if 'Importance' in explainer_names and not model_supports_importance:
            explainerlist.remove('Importance')
         
        feature_importances_matrix = np.zeros((len(df), len(feature_names) * len(explainerlist)))
        results = []
        for  i in tqdm(df.index, desc="Computing importance"):
            result = train_model._explain_point_model(df, feature_names, target_name, i, weights, trained_models, explainerlist,skleanrmodel)
            results.append(result)
        for i, importances in enumerate(results):
            feature_importances_matrix[i, :] = importances
        columns = [var + '_' + explainer for explainer in explainerlist  for var in feature_names]
        importances_df = pd.DataFrame(feature_importances_matrix, columns=columns)
        df_new = pd.concat([df.reset_index(drop=True), importances_df.reset_index(drop=True)], axis=1)
        return df_new  


    def _point_partial_dependent_model(df, point_index, sample_var, trained_model,  feature_names, target_name):
        X_test1 = df[feature_names].iloc[point_index].values.reshape(1, -1)
        model = trained_model
        y_pred_matrix = np.zeros((sample_var.shape[0], len(feature_names) + 1))
        for i, feature_name in enumerate(feature_names):
            for j, val in enumerate(sample_var[feature_name]):
                X_test = X_test1.copy()
                X_test[0, i] = val
                y_pred = model.predict(X_test)[0]
                y_pred_matrix[j, i] = y_pred
        y_pred_matrix[:, -1] = range(sample_var.shape[0])
        return y_pred_matrix
        
    def partial_dependence_model(df, trained_models, feature_names, target_name, weights,num_samples=50,even=False):
        if even==False:
            sampled_df, sample_var=preprocess._generate_samples(df,feature_names, num_samples)
        else:
            sampled_df, sample_var=preprocess._generate_even_samples(df,feature_names, num_samples)
        print(f'Sample size: {num_samples}. {len(sampled_df)} used for Partial Dependence Estimation')
        results=[]
        for i in tqdm(sampled_df.index, desc="Computing importances"):
            result=train_model._point_partial_dependent_model(df, i ,sample_var, trained_models[i],  feature_names, target_name)
            results.append(result) 
        merged_array = np.vstack(results)
        feature_id=feature_names.copy()
        feature_id.append('ID')
        resultsdf=pd.DataFrame(merged_array, columns=feature_id)
        agg_dict = {col: 'mean' for col in feature_id}
        results_mean = resultsdf.groupby('ID').agg(agg_dict)
        results_mean.drop(columns=['ID'],inplace=True)
        results_mean.columns = [f"{col}_estimate" for col in results_mean.columns]
        sample_var.columns= [f"{col}_sample" for col in sample_var.columns]
        df_pd= pd.concat([sample_var.reset_index(drop=True), results_mean.reset_index(drop=True)], axis=1)
        return  df_pd  

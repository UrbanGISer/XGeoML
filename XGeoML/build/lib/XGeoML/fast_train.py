import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from .preprocess import *

# 03 Quick train and predict without keeping models
class fast_train:
    # predict
    def _predict_point(df, feature_names, target_name, point_index, weights, model_class):
        X_train, y_train, X_test = preprocess._preprocess_data(df, feature_names, target_name, point_index, weights)
        model = model_class()  # Assuming model_class is defined elsewhere
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        return y_pred 
        
    def predict(df, feature_names, target_name, weights, model_class):
        predictions = np.zeros(len(df))
        results = Parallel(n_jobs=-1)(
            delayed(fast_train._predict_point)(df, feature_names, target_name, i, weights, model_class) for i in tqdm(range(len(df)), desc="Computing Prediction")
        )
        for i, y_pred in enumerate(results):
            predictions[i] = y_pred
        df['predy'] = predictions
        return  df     
     
    # predict and explain  
    def _explain_point(df, feature_names, target_name, point_index, weights, model_class, explainer_names):
        X_train, y_train, X_test =preprocess._preprocess_data(df, feature_names, target_name, point_index, weights)
        model = model_class()  # Assuming model_class is defined elsewhere
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        importances = {}
        if 'LIME' in explainer_names:
            importances['LIME'] = preprocess._calculate_lime_importances(model, X_train, X_test, feature_names, target_name)
        if 'SHAP' in explainer_names:
            importances['SHAP'] = preprocess._calculate_shap_importances(model, X_train, X_test)            
        if 'Importance' in explainer_names and hasattr(model, 'feature_importances_'):
            importances['Importance'] = model.feature_importances_
        importance_combine = np.concatenate([value.reshape(-1) for value in importances.values()])
        return y_pred, importance_combine

    def explain(df, feature_names, target_name, weights, model_class, explainer_names):
        #pretest
        X_train1, y_train1, _ =preprocess._preprocess_data(df, feature_names, target_name, 1, weights)
        model = model_class()  # Assuming model_class is defined elsewhere
        model.fit(X_train1, y_train1)
        model_supports_importance = hasattr(model, 'feature_importances_')
        explainerlist= explainer_names.copy()
        if 'Importance' in explainer_names and not model_supports_importance:
            explainerlist.remove('Importance')
        predictions = np.zeros(len(df))
        feature_importances_matrix = np.zeros((len(df), len(feature_names) * len(explainerlist)))
        results = Parallel(n_jobs=-1)(
            delayed(fast_train._explain_point)(df, feature_names, target_name, i, weights, model_class, explainerlist) for i in tqdm(range(len(df)), desc="Computing Prediction")
        )
        for i, (y_pred, importances) in enumerate(results):
            predictions[i] = y_pred
            feature_importances_matrix[i, :] = importances
        columns = [var + '_' + explainer for explainer in explainerlist for var in feature_names]
        importances_df = pd.DataFrame(feature_importances_matrix, columns=columns)
        importances_df['predy'] = predictions
        df_new = pd.concat([df.reset_index(drop=True), importances_df.reset_index(drop=True)], axis=1)
        return df_new
     
    def _point_partial_dependent(df, point_index, sample_var, model_class,  feature_names, target_name, weights):
        X_train, y_train, X_test1 = preprocess._preprocess_data(df, feature_names, target_name, point_index, weights)
        tmodel = model_class()
        tmodel.fit(X_train, y_train)
        y_pred_matrix = np.zeros((sample_var.shape[0], len(feature_names) + 1))
        for i, feature_name in enumerate(feature_names):
            for j, val in enumerate(sample_var[feature_name]):
                X_test = X_test1.copy()
                X_test[0, i] = val
                y_pred = tmodel.predict(X_test)[0]
                y_pred_matrix[j, i] = y_pred
        y_pred_matrix[:, -1] = range(sample_var.shape[0])

        return y_pred_matrix
        
    def partial_dependence(df, model_class, feature_names, target_name, weights,num_samples=50,even=False):
        if even==False:
            sampled_df, sample_var=preprocess._generate_samples(df,feature_names, num_samples)
        else:
            sampled_df, sample_var=preprocess._generate_even_samples(df,feature_names, num_samples)
        print(f'Sample size: {num_samples}. {len(sampled_df)} used for Partial Dependence Estimation')
        results = Parallel(n_jobs=4)(delayed(fast_train._point_partial_dependent)(df, i, sample_var, model_class,  feature_names, target_name, weights) for i in tqdm(sampled_df.index, desc="Computing importances"))
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
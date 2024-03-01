import numpy as np
import lime.lime_tabular
import shap
import pandas as pd

# 02 Define preprocess data function and explainer
class preprocess:
    def _preprocess_data(df, feature_names, target_name, point_index, weights):
        weight = weights[point_index]
        X_weighted = df[feature_names].multiply(weight, axis=0)
        y_weighted = df[target_name] * weight
        non_zero_indices = np.nonzero(weight)[0]
        selected_points = non_zero_indices[non_zero_indices != point_index]
        X_train = X_weighted.iloc[selected_points].values
        y_train = y_weighted.iloc[selected_points]
        current_point = X_weighted.iloc[point_index].values.reshape(1, -1)
        return X_train, y_train, current_point

    # Explainer
    def _calculate_lime_importances(model, X_train, X_test, feature_names, target_name):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=[target_name],
            mode='regression',
            discretize_continuous=False,
        )
        exp = explainer.explain_instance(data_row=X_test[0], predict_fn=model.predict)
        lime_importances_dict = dict(exp.as_list())
        importances = np.array([lime_importances_dict.get(feature, 0) for feature in feature_names])
        return importances

    def _calculate_shap_importances(model, X_train, X_test):
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        return np.abs(shap_values.values).mean(axis=0)

    # Partial Dependence Estimation
    # original sample
    def _generate_col_samples(df, col, num_samples):
        ranked_values = df[col].rank(method='first')
        min_rank, max_rank = ranked_values.min(), ranked_values.max()
        normalized_ranks = ((ranked_values - min_rank) / (max_rank - min_rank) * (num_samples - 1) + 1).astype(int)
        col_samples = pd.Series(index=range(1, num_samples + 1), dtype=np.float64)
        sampled_df = pd.DataFrame()
        for rank in range(1, num_samples + 1):
            subset = df[normalized_ranks == rank]
            if not subset.empty:
                sampled_row = subset.sample(n=1)
                sampled_df = pd.concat([sampled_df, sampled_row])
                col_samples[rank] = sampled_row[col].iloc[0]
        return sampled_df, col_samples

    def _generate_samples(df, feature_names, num_samples):
        sample_df = pd.DataFrame()
        sample_var = pd.DataFrame(index=range(1, num_samples + 1))
        for col in feature_names:
            col_df, col_samples = preprocess._generate_col_samples(df, col, num_samples)
            sample_df = pd.concat([sample_df, col_df])
            sample_var[col] = col_samples
        sample_df.drop_duplicates(inplace=True)
        sampled_df=sample_df[feature_names]
        return sampled_df, sample_var  

    # even sample  
    def _generate_even_samples(df,feature_names, num_samples):
        sampled_df = pd.DataFrame()
        sample_var = pd.DataFrame()
        for col in feature_names:
            min_val, max_val = df[col].min(), df[col].max()
            samples = np.linspace(min_val, max_val, num_samples)
            for val in samples:
                subset = df[(df[col] >= val) & (df[col] < val + (max_val - min_val) / num_samples)]
                if not subset.empty:
                    sampled_df = pd.concat([sampled_df, subset.sample(n=1)])
            sample_var[col] = samples
        sampled_df.drop_duplicates(inplace=True)
        sampled_df=sampled_df[feature_names]
        return sampled_df, sample_var

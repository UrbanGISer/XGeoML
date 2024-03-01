# An Ensemble Framework for Geospatial Machine Learning

 
GitHub: https://github.com/UrbanGISer/XGeoML

PYPI Homepage: https://pypi.org/project/XGeoML/0.1.4/

Installation: pip install XGeoML

This package addresses the critical challenge of analyzing and interpreting spatially varying effects in geographic analysis, stemming from the complexity and non-linearity of geospatial data. We introduce an innovative integrated framework that combines local spatial weights, Explainable Artificial Intelligence (XAI), and advanced machine learning technologies. This approach significantly bridges the gap between traditional geographic analysis models and contemporary machine learning methodologies.

## Introduction

Geospatial data is inherently complex and non-linear, presenting significant challenges in analysis and interpretation. Traditional geographic analysis models often struggle to address these challenges, leading to gaps in understanding and interpretation.

## Our Approach

We propose an innovative integrated framework that leverages local spatial weights, Explainable Artificial Intelligence (XAI), and advanced machine learning technologies. Our approach aims to bridge the gap between traditional methods and modern machine learning techniques, offering a more comprehensive tool for geographic analysis.

### Features

- **Local Spatial Weights:** Incorporates the spatial context of data, enhancing model sensitivity to geographical nuances.
- **Explainable Artificial Intelligence (XAI):** Provides clarity on the decision-making process, improving the interpretability of the model's predictions.
- **Advanced Machine Learning Technologies:** Utilizes cutting-edge algorithms to manage the complexity and non-linearity of geospatial data effectively.

## Key Functions
- **Use built-in Spatial Weights:** Generate Gaussian, Binary and GaussianBinary weight.
```python
weights=w_matrix.spatial_weight(df, "u", "v", fix=False, bandwidth=80, kernel_type='Binary')
```
- **Import libpysal Spatial Weights:** Accept all spatial weight.
```python
import libpysal.weights as lw
points = df[['u', 'v']].values
w=lw.DistanceBand(points,threshold=6,binary=False)
weightpysal=w_matrix.from_libpysal(w)
```
- **Predict or Search Bandwidth with fast training model:** Accept all sci-learn model.
```python
# 01 Define key variables
feature_names=['x1','x2','x3','x4']
target_name="y"
explainer_names = ["LIME","SHAP", 'Importance']
turebeta= ['b_linear','b_circular', 'b_cos_basic',  'b_poly']

# 02 import  sklearn ML model
from sklearn.ensemble import  GradientBoostingRegressor
model=GradientBoostingRegressor

# 03 import  R2
from sklearn.metrics  import r2_score

# 04 generate weights
weights=w_matrix.spatial_weight(df, "u", "v", fix=False, bandwidth=80, kernel_type='Binary')

# 05 Bandwidth Searching
eval_bandwidth = pd.DataFrame()
for i in range(10):
    k=40+i*40
    for j in range(3):
        weights=w_matrix.spatial_weight(df, "u", "v", fix=False, bandwidth=k, kernel_type='Binary')
        dfx=fast_train.predict(df, feature_names, target_name, weights, model)
        r22=r2_score(dfx.y,dfx.predy)
        eval_bandwidth.loc[i, j]=r22
```

- **Predict and Evaluate with updated Spatial Weights:** Using new spatial weight based on bandwidth searching.
```python
# 06 Predict
df_pred=fast_train.predict(df, feature_names, target_name, weights, model)
# 07 Evaluate
from sklearn.metrics import r2_score
r2_score(df_pred.y,df_pred.predy)
```

- **Explain models:** explainer_names must be in a list["LIME","SHAP", 'Importance'].
```python
# 08 Explain
df_explain=fast_train.explain(df, feature_names, target_name, weights,model, explainer_names)
```
- **Partial Dependence Estimation:** Sample bin is used here, two mode: even or original values.
```python
# 09 Partial dependence
df_pd=fast_train.partial_dependence(df, model,  feature_names, target_name, weights,num_samples=50,even=False)
```

- **Use trained models:** MUST BE CAREFUL, It might be time consuming while use HyperOpt.
```python
#10 Trained models
sk_models,predictions=train_model.train_sklearn(df, feature_names, target_name, weights, model)
# 11 Explain with trained sci-learn models
df_sk=train_model.explain_models(df, feature_names, target_name, weights, sk_models, explainer_names)
#12 PDE with trained sci-learn models
df_sk_pd_even=train_model.partial_dependence_model(df, sk_models, feature_names, target_name, weights,num_samples=50)

# 13 Explain with trained HyperOpt models: SUPER TIME CONSUMING
from hpsklearn import HyperoptEstimator, xgboost_regression, mlp_regressor
from hyperopt import tpe
hymodel=xgboost_regression
#max_eval=5 for 900 points, it takes 3 hours
hy_models,predictions=train_model.train_hysklearn(df, feature_names, target_name, weights, hymodel,max_evals=1, trial_timeout=60)

# 14 Explain with trained models. MUST set  skleanrmodel=False
df_hy=train_model.explain_models(df, feature_names, target_name, weights, hy_models, explainer_names,skleanrmodel=False)
# 14 Partial dependence with trained models. Same as Previous one
df_hy_pd_even=train_model.partial_dependence_model(df, hy_models, feature_names, target_name, weights,num_samples=50)
```

Through rigorous testing on synthetic datasets and real-world dataset, our framework has proven to enhance the interpretability and accuracy of geospatial predictions in both regression and classification tasks. It effectively elucidates spatial variability, representing a significant advancement in the precision of predictions and offering a novel perspective for understanding spatial phenomena.

## Conclusion

Our integrated framework marks a significant step forward in geographic analysis. By combining local spatial weights, XAI, and advanced machine learning, we offer a powerful tool for analyzing and interpreting complex geospatial data. This approach not only improves the accuracy and interpretability of geospatial predictions but also provides a fresh perspective on spatial phenomena.

## Contact

For further information, inquiries, or collaborations, please contact us at [lingboliu@fas.harvard.edu](mailto:lingboliu@fas.harvard.edu).


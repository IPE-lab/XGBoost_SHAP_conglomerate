# **XGBoost_SHAP_conglomerate**
## Running screenshots show

  <img src="img/SHAP values of training set.jpg" width="400" />
- **SHAP values of training set**
  <img src="img/Mean SHAP value of input factors0.jpg" width="400" />
  <img src="img/Mean SHAP value of input factors1.jpg" width="400" />
- **Mean SHAP value of input factors**
- ***
## Paper Support
- Original information: Investigation of hydraulic fracture propagation in conglomerate rock using discrete element method and explainable machine learning framework
- Recruitment Journal: Acta Geotech
- Original DOI: https://doi.org/10.1007/s11440-024-02317-9
***
## Description of the project
As an important unconventional oil and gas resource, conglomerate reservoirs are characterized by low porosity and low permeability, and are commonly stimulated by large-scale hydraulic fracturing technology. However, its strong inhomogeneity will induce irregular propagation of hydraulic fractures, which will increase the difficulty of fracture morphology prediction and the deployment of fracturing schemes. Most of the traditional studies on hydraulic fracture propagation in conglomerates show qualitative analysis results, which are unable to predict the actual working conditions. This method applies the explainable machine learning framework (XGBoost-SHAP) to address this shortcoming, so as to realize the accurate prediction of fracture morphology and to provide the interpretation for the prediction results. The training dataset are constructed by curating data from numerical simulation and existing studies. 
***
## Functions of the project
XGBoost_SHAP_conglomerate. py is used for prediction and explanation of the fracture morphology. It contains three functions.
1. Train and test the XGBoost model on the split dataset, and report the test results
    train(X, y, random_seed):
2. Import the datafile, set the model training and get explanation across the entire dataset
    main(datafile_path, seed):
3. Predict and explain single or several study cases from manual input
    case_explain(classifier, explainer, study_case):
***
## The operating environment of the project
-	Python == 3.10.9
-	pandas == 1.5.3
-	xgboost == 1.7.5
-	shap == 0.41.0
-	matplotlib == 3.6.3
-	numpy == 1.23.5
-	scikit-learn== 1.2.1
***
## How to use the project
#### 1、Prepare the training data as tabular data, store them in a .csv (or .xlsx) file, which should contain 5 columns (Strength ratio, Interface permeability, Sum of stress difference, Viscosity&Injection rate and Target value)

#### 2、Fill in the right path to the data file, and run XGBoost_SHAP_conglomerate. py

***

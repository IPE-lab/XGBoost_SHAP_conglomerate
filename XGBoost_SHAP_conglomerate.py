import pandas as pd
import xgboost as xgb
import shap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def train(X, y, random_seed):
    total_loss = 0
    n=0
    for random_state in range(random_seed, random_seed+10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
        classifier = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_test = np.array(y_test)
        loss = np.abs(y_test-y_pred).sum()
        total_loss = total_loss+loss
        n= n+1
        print(n)
        print("truth:", y_test)
        print("pred:", y_pred)
    print("Correct Prediction:", 10*y_test.shape(-1)-total_loss,"Wrong Prediction:", total_loss, "Total Prediction:", 10*y_test.shape(-1), "Accuracy:", (10*y_test.shape(-1)-total_loss)/(10*y_test.shape(-1)))


def main(datafile_path, seed):
    df = pd.read_excel(datafile_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    train(X, y, seed)

    #train classifier on all data to get the SHAP values
    classifier = xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200)
    classifier.fit(X, y)

    explainer = shap.Explainer(classifier)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X)
    plt.savefig('shap_summary_plot.png')

    plt.figure()
    shap.summary_plot(shap_values, plot_type="bar")
    plt.savefig('shap_bar_plot.png')

    return classifier, explainer



def case_explain(classifier, explainer, study_case):
    study_case_pred = classifier.predict(study_case)
    shap_values = explainer(study_case)

    plt.figure()
    shap.plots.force(shap_values[0], feature_names=study_case.columns, matplotlib=True, figsize=(20, 5), text_rotation=10)
    plt.savefig('shap_force_plot.png')




if __name__ == "__main__":
    DATAFILE_PATH = 'path/to/datafile.csv'
    SEED = 40

    classifier, explainer = main(DATAFILE_PATH, SEED)

    #If you want to prdict and explain a single case
    study_case = pd.DataFrame({'Strength ratio': [1.0],
                            'Interface permeability': [200],
                            'Sum of stress differencce': [15],
                            'Viscosity&Injection rate': [360]})
    case_explain(classifier, explainer, study_case)
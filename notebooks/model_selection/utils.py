from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def svm_classifier (X_train, y_train, X_test):
    # Train the model
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Predict the label for the test data
    y_pred = svm.predict(X_test)

    return y_pred

def gradient_boosting_classifier (X_train, y_train, X_test):
    # Train a Gradient Boosting classifier
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = xgb_clf.predict(X_test)

    return y_pred

def naive_bayes_classifier (X_train, y_train, X_test):
    # Train a Naive Bayes classifier
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = nb_clf.predict(X_test)
    # Evaluate using accuracy, precision, recall, F1-score as before

    return y_pred

def mlp_classifier(X_train, y_train, X_test):

    if isinstance(X_train, csr_matrix):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train

    if isinstance(X_test, csr_matrix):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test

    mlp = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', 
                        max_iter=35, batch_size=128, verbose=True)

    mlp.fit(X_train_dense, y_train)

    # Predict
    y_pred = mlp.predict(X_test_dense)

    return y_pred

def logistic_regression_classifier(X_train, y_train, X_test):
    # Train a logistic regression classifier
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = lr_clf.predict(X_test)

    return y_pred

def update_performance(df, model_name, y_test, y_pred):

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract weighted average metrics
    weighted_avg = report['weighted avg']
    precision = weighted_avg['precision']
    recall = weighted_avg['recall']
    f1_score = weighted_avg['f1-score']

    # Create a new DataFrame for the row to be added
    new_row_df = pd.DataFrame({'Model': [model_name], 
                               'Precision': [precision], 
                               'Recall': [recall], 
                               'F1-Score': [f1_score]})

    # Concatenate the new row with the existing DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    return df

def plot_performance(performance_df):    
    # Plotting
    n_models = len(performance_df)
    ind = np.arange(n_models)  # the x locations for the groups
    width = 0.25  # the width of the bars
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting each metric
    rects1 = ax.bar(ind - width, performance_df['Precision'], width, label='Precision')
    rects2 = ax.bar(ind, performance_df['Recall'], width, label='Recall')
    rects3 = ax.bar(ind + width, performance_df['F1-Score'], width, label='F1-Score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Performance by Model and Metric')
    ax.set_xticks(ind)
    ax.set_xticklabels(performance_df['Model'])
    ax.legend()

    # Attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Call the function to attach the labels
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Show the plot
    plt.show()
    
def classify_model(X_train , tokenizer , model) :
    preds = []
    preds_proba = []
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
    for x in X_train:
        with torch.no_grad():
            input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
            logits = model(**input_sequence).logits
            scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        label = max(scores, key=scores.get)
        probability = max(scores.values())
        preds.append(label)
        preds_proba.append(probability)

    return preds, preds_proba
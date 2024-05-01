"""
Michael Moreno
Class: CS 677 - Spring 2
Date: 04/25/2024
Final Project
Create a dataframe from the results.csv file and add a class label column based on whether
a team got a positive or negative result. Use the logistic regression model to train the 
data and predict values. Compute the accuracy and confusion matrix.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

results_file = 'results.csv'

with open(results_file) as f:
    print(f'opened file for {results_file}.')
    original_df = pd.read_csv(results_file)
    original_df['date'] = pd.to_datetime(original_df['date'])
    # Creates a new dataframe that includes matches from the 21st century where there are no neutrally played games
    post_2000_df = original_df[(original_df['date'] > pd.Timestamp('2000-01-01')) & (~original_df['neutral'])]
    post_2000_df.to_csv('post_2000_results.csv')
    post_2000_df = post_2000_df.copy()

    pos_result_label_list = []

    home_score_column = post_2000_df['home_score']
    away_score_column = post_2000_df['away_score']

    for index,row in post_2000_df.iterrows():
        if row['home_score'] >= row['away_score']:
            pos_result_label_list.append(1)
        elif row['home_score'] < row['away_score']:
            pos_result_label_list.append(0)


    post_2000_df['Class Label'] = pos_result_label_list

    encoder = OneHotEncoder(handle_unknown='ignore')

    X_encoded = encoder.fit_transform(post_2000_df[['home_team', 'away_team', 'home_score', 'away_score']])

    X_numeric = post_2000_df[['home_score', 'away_score']].values
    X_numeric = X_numeric.reshape(-1, 2)
    X_encoded_dense = X_encoded.toarray()
    X = np.concatenate((X_numeric, X_encoded_dense), axis=1)
    y = post_2000_df['Class Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()

    accuracy = accuracy_score(y_test, y_pred)
    formatted_accuracy = "{:.2f}".format(accuracy)
    print(f'Logistic Regression Accuracy: {formatted_accuracy}')

    def calculate_pos_neg(true_list, pred_list):
        """A function that calculates the number of true positive, false positive, true
        negative, and false negative values."""
        true_pos_count = 0
        false_pos_count = 0
        true_neg_count = 0
        false_neg_count = 0
        for i in range(len(true_list)):
            if true_list[i] == 0 and int(pred_list[i]) == 0:
                true_pos_count += 1
            elif true_list[i] == 1 and int(pred_list[i]) == 0:
                false_pos_count += 1
            elif true_list[i] == 1 and int(pred_list[i]) == 1:
                true_neg_count += 1
            elif true_list[i] == 0 and int(pred_list[i]) == 1:
                false_neg_count += 1
        return true_pos_count, false_pos_count, true_neg_count, false_neg_count
    
    log_reg_model = calculate_pos_neg(y_test_list, y_pred_list)
    print(f"Logistic Regression Confusion Matrix: {log_reg_model}")

    def calculate_tpr(true_pos, false_neg):
        """ A function that calculates the true positive rate, or the percentage of positive values that
        we predicted correctly."""
        calculation = true_pos/(true_pos + false_neg)
        formatted_calculation = "{:.2%}".format(calculation)
        return formatted_calculation

    def calculate_tnr(true_neg, false_pos):
        """ A function that calculates the true negative rate, or the percentage of negative values that
         we predicted correctly."""
        calculation = true_neg/(true_neg + false_pos)
        formatted_calculation = "{:.2%}".format(calculation)
        return formatted_calculation
    
    log_reg_tpr = calculate_tpr(log_reg_model[0], log_reg_model[3])
    print(f"Logistic Regression TPR: {log_reg_tpr}")

    log_reg_tnr = calculate_tnr(log_reg_model[2], log_reg_model[1])
    print(f"Logistic Regression TNR: {log_reg_tnr}")

    confusion_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Logistic Regression Confusion Matrix')
    plt.savefig('Logistic Regression Confusion Matrix.pdf')

"""
Michael Moreno
Class: CS 677 - Spring 2
Date: 04/25/2024
Final Project
Create a dataframe from the results.csv file and add a class label column based on whether
a team got a positive or negative result. Use the decision trees model to find the best 
value for the parameter max_depth. Then, train the data with the decision trees model 
using the best value for max_depth. Compute the accuracy and confusion matrix.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
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

    # Combine the encoded columns with the numeric columns
    X_numeric = post_2000_df[['home_score', 'away_score']].values
    X_numeric = X_numeric.reshape(-1, 2)
    X_encoded_dense = X_encoded.toarray()
    X = np.concatenate((X_numeric, X_encoded_dense), axis=1)
    y = post_2000_df['Class Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    max_depths = range(1, 21)
    max_accuracy = 0
    best_max_depth = None
    accuracies = []

    for max_depth in max_depths:
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_max_depth = max_depth

    print(f"Best max_depth: {best_max_depth}, Max accuracy: {max_accuracy}")

    plt.plot(max_depths, accuracies, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs max_depth')
    plt.grid(True)
    plt.savefig('Decision Tree Accuracy vs max_depth.pdf')

    model = DecisionTreeClassifier(max_depth=best_max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test_list = y_test.tolist()
    y_pred_list = y_pred.tolist()

    accuracy = accuracy_score(y_test, y_pred)
    formatted_accuracy = "{:.2f}".format(accuracy)
    print(f"Decision Trees Accuracy with max_depth = {best_max_depth}: {formatted_accuracy}.")

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
    
    dt_model = calculate_pos_neg(y_test_list, y_pred_list)
    print(dt_model)

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
    
    dt_tpr = calculate_tpr(dt_model[0], dt_model[3])
    print(f"Decision Trees with max_depth = {best_max_depth}: TPR: {dt_tpr}")

    dt_tnr = calculate_tnr(dt_model[2], dt_model[1])
    print(f"Decision Trees with max_depth = {best_max_depth}: TNR: {dt_tnr}")

    confusion_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Decision Tree Confusion Matrix')
    plt.savefig('Decision Tree Confusion Matrix.pdf')

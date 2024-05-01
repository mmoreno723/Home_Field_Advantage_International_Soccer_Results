"""
Michael Moreno
Class: CS 677 - Spring 2
Date: 04/25/2024
Final Project
Create a dataframe from the results.csv file and add a class label column based on whether
a team got a positive or negative result. Use the random forests model to find the best 
values for the parameters number of subtrees and max_depth. Then, train the data with the 
random forests model using the best values for number of subtrees and max_depth. Compute 
the accuracy and confusion matrix.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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
    X = encoder.fit_transform(post_2000_df[['home_team', 'away_team', 'home_score', 'away_score']])
    y = post_2000_df['Class Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    def test_hyperparameters(N,d):
        rf = RandomForestClassifier(n_estimators=N, max_depth=d, random_state=0)
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        return 1-accuracy_score(y_test,y_pred)
    
    errors = np.empty([10,5])

    for N in range(1,11):
        for d in range(1,6):
            errors[N-1][d-1] = test_hyperparameters(N,d)

    error_rates = []
    n_estimators_values = [1, 2, 3, 4, 5]

    best_N_index, best_d_index = np.unravel_index(np.argmin(errors), errors.shape)
    best_N = best_N_index + 1
    best_d = best_d_index + 1
    best_error_rate = errors[best_N_index][best_d_index]
    formatted_best_error_rate = "{:.2f}".format(best_error_rate)

    print(f"Optimal values: N = {best_N}, d = {best_d}, Error Rate = {formatted_best_error_rate}")

    df = pd.DataFrame(errors, columns = ['Depth 1','Depth 2','Depth 3','Depth 4','Depth 5'], index = range(1,11))
    df.plot()
    plt.xlabel("N estimators")
    plt.ylabel("Error")
    plt.title("Errors by number of estimators for max_depth in {1,2,3,4,5}")
    plt.savefig("Random Forests errors by # estimators for max_depth.pdf")

    rf_best = RandomForestClassifier(n_estimators=best_N, max_depth=best_d, random_state=0)

    rf_best.fit(X_train, y_train)

    y_pred_best = rf_best.predict(X_test)

    y_test_list = y_test.tolist()
    y_pred_best_list = y_pred_best.tolist()

    accuracy_best = accuracy_score(y_test, y_pred_best)
    formatted_accuracy_best = "{:.2f}".format(accuracy_best)
    print(f"Accuracy of the model using N = {best_N}, d = {best_d}: {formatted_accuracy_best}")

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
    
    best_n_d_model = calculate_pos_neg(y_test_list, y_pred_best_list)
    print(f"Random Forests N = {best_N}, d = {best_d} Confusion Matrix: {best_n_d_model}")

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
    
    best_n_d_tpr = calculate_tpr(best_n_d_model[0], best_n_d_model[3])
    print(f"N = {best_N}, d = {best_d} TPR: {best_n_d_tpr}")

    best_n_d_tnr = calculate_tnr(best_n_d_model[2], best_n_d_model[1])
    print(f"N = {best_N}, d = {best_d} TNR: {best_n_d_tnr}")

    confusion_mat = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Random Forest N = 3, d = 5 Confusion Matrix')
    plt.savefig('Random Forest N = 3, d = 5 Confusion Matrix.pdf')

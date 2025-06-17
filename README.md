
# Bank Marketing Campaign Analysis

This project aims to predict whether a client will subscribe to a bank long-term deposit product based on telemarketing campaign data. The analysis follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## Table of Contents

1.  [Business Understanding](#1-business-understanding)
2.  [Data Understanding](#2-data-understanding)
3.  [Data Preparation](#3-data-preparation)
4.  [Modeling](#4-modeling)
5.  [Evaluation](#5-evaluation)
6.  [Addressing Class Imbalance and Model Optimization](#6-addressing-class-imbalance-and-model-optimization)
7.  [Next Steps and Recommendations](#7-next-steps-and-recommendations)



## 1. Business Understanding

**Objective:** The primary goal is to predict if a client will subscribe (binary: 'yes'/'no') to a bank long-term deposit, based on their interaction with telemarketing campaigns. This predictive model will help the Portuguese banking institution optimize future marketing campaigns by identifying potential subscribers more effectively.



**Business Problem:** Enterprises promote products through mass campaigns or directed marketing. This project focuses on directed marketing, where a Portuguese bank used its contact center for telemarketing campaigns. The core business question is to improve the efficiency of these campaigns by accurately predicting customer subscription behavior.

**Campaign Flow:**
  ![Process Flow ](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/Process_flow.png)


## 2. Data Understanding

The dataset is sourced from the UCI Machine Learning repository. It's related to direct marketing campaigns of a Portuguese banking institution. The data is a collection of results from multiple marketing campaigns conducted via telephone.

*   **Bank Marketing Campaign Dataset:** [Link to Bank Marketing Data File](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/tree/main/data/bank-additional-full.csv) 
*   **Jupyter Notebook:** [Link to Jupyter Notebook](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/tree/main/Bank_Marketing_Campaign_PA_III.ipynb) 
*   **Accompanying Article CRISP DM:** [Link to CRISP DM](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/tree/main/CRISP-DM-BANK.pdf) (For more details on data and features)

**Initial Data Inspection:**

*   **Shape:** (41188 rows, 21 columns)
*   **Missing Values:** No explicit missing values (all columns show 41188 non-null counts). However, unknown values are present in some categorical features, which will be treated during data preparation.
*   **Data Types:** A mix of numerical (int64, float64) and categorical (object) types.
    *   Numeric: 10
    *   Categorical: 10
    *   Boolean: 1 (after y is mapped)

**EDA Report:**

*   A complete summary of data analysis details can be found in the [Link to EDA Report file](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/tree/main/bank_marketing_eda_report.html) file.



**Key Features and Target Variable:**

*   **Bank Client Data:** age, job, marital, education, default (credit in default), housing (housing loan), loan (personal loan).
*   **Last Contact of Current Campaign:** contact (communication type), month, day\_of\_week, duration (last contact duration in seconds).
    *   **Important Note on `duration`:** This attribute highly affects the output target. However, duration is not known before a call is performed, and y is known after the call. Therefore, duration should be excluded if the intention is to build a realistic predictive model, and only included for benchmark purposes.
*   **Other Attributes:** campaign (number of contacts for this client), pdays (days since last contact from a previous campaign; 999 means not previously contacted), previous (number of contacts before this campaign), poutcome (outcome of previous marketing campaign).
*   **Social and Economic Context Attributes:** emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed.
*   **Output Variable (Target):** y - has the client subscribed a term deposit? (binary: 'yes'/'no'). This column is renamed to `deposit` in the notebook for clarity.

**Target Variable Distribution (`deposit`):**

*   no: 36548 (Approx. 88.7%)
*   yes: 4640 (Approx. 11.3%)

**Observation:** Significant class imbalance, with the 'no' class being dominant. This will need to be addressed during modeling to avoid biased predictions towards the majority class.

**Initial Alerts/Observations from EDA (via ydata-profiling):**

*   **Duplicate Rows:** 12 duplicate rows found (less than 0.1%).
*   **High Correlation:** Several features are highly correlated, for example:
    *   cons.conf.idx with month
    *   cons.price.idx with contact, emp.var.rate, month
    *   contact with cons.price.idx, month, nr.employed
    *   emp.var.rate with cons.price.idx, euribor3m, month, nr.employed
    *   euribor3m with emp.var.rate, month, nr.employed
    *   housing with loan
    *   loan with housing
    *   month with multiple social/economic context attributes and contact.
    *   nr.employed with contact, emp.var.rate, euribor3m, month.
    *   pdays with poutcome, previous.
    *   poutcome with pdays, previous.
    *   previous with pdays, poutcome.
*   **Imbalance:** default (53.3% imbalanced, referring to 'unknown' being a large category), loan (51.3% imbalanced, 'no' being dominant), poutcome (56.8% imbalanced, 'nonexistent' being dominant).
*   **Zeros:** previous has a high percentage of zeros (86.3%), indicating many clients were not previously contacted.


**Visualizations:**

To better understand the data, several visualizations were created to explore the relationships between different features and the target variable. Here are some key observations from these visualizations:

*   **Deposit Subscription by Education Level:**

    ![Education vs. Deposit](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/education_vs_deposit.png)

*   **Deposit Subscription by Marital Status:**

    ![Marital vs. Deposit](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/marital_vs_deposit.png)

*   **Deposit Subscription by Job Type:**

    ![Job vs. Deposit](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/job_vs_deposit.png)

*   **Deposit Subscription by Contact Type:**

    ![Contact vs. Deposit](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/contact_vs_deposit.png)

These visualizations provide insights into how customer attributes correlate with deposit subscriptions. For instance, the "Education vs. Deposit" plot shows the distribution of deposit subscriptions (yes/no) across different categories:

*   **Distribution of Education by Deposit Subscription:** Shows that "university.degree" has the highest number of deposit subscriptions, followed by "high.school". "basic.9y" and "basic.4y" have lower subscription rates.
*   **Distribution of Marital by Deposit Subscription:** "married" individuals have the highest number of "no" subscriptions, while "single" individuals show a notable number of "yes" subscriptions.
*   **Distribution of Job by Deposit Subscription:** "admin." and "blue-collar" jobs have the highest overall counts. "admin." and "technician" show relatively higher "yes" subscriptions compared to "blue-collar" and "services".
*   **Distribution of Contact by Deposit Subscription:** This chart is less clear as the labels are obscured, but it appears to compare different contact methods.

These plots help to visualize customer demographics (education, marital status, job, and contact method) and their correlation with deposit subscriptions. Key highlights include the strong performance of university graduates and individuals with administrative or technician jobs in terms of deposit subscriptions.


*   **Loan vs. Deposit:**

    ![Loan vs. Deposit](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/Loan_vs_deposit.png)

*   **Housing Loan:** Among those who subscribed to a deposit, 54.0% had a housing loan, 43.7% did not, and 2.3% were unknown.
*   **Loan (Personal Loan):** For deposit subscribers, 83.0% did not have a personal loan, 14.7% did, and 2.3% were unknown.
*   **Default Loan:** A large majority (90.5%) of deposit subscribers had no default loan, while a small percentage (9.5%) had an unknown default status. None had a default loan


*   **Deposit Acceptance by Day of Week, Month and Contact Type**

    ![Contact vs. Deposit](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/Deposit_Acceptance_Week_Month_Type.png)

**Summary for Deposit Acceptance ('yes') by Day of Week:**

Deposit acceptance is distributed relatively evenly across the days of the week.
Thursday has the highest acceptance rate at 22.3%.
Wednesday and Tuesday show similar acceptance rates at 20.5% each.
Monday is slightly lower at 18.3%, and Friday at 18.2%.

**Deposit Acceptance ('yes') by Month:**

May accounts for the largest share of deposit acceptances at 19.1%.
April and August are also significant, each contributing 11.6% and 14.1% respectively.
December and July are close behind with 13.9% and 12.0% respectively.
September (5.5%) and June (5.9%) show the lowest acceptance rates among the months displayed.
October and November contribute 6.8% and 9.0% respectively.

**Deposit Acceptance ('yes') by Contact Type:**

The vast majority of deposit acceptances come through 'cellular' contact, accounting for 83.0%.
'Telephone' contact accounts for a much smaller portion, at 17.0%.

**Overall Summary:**

The data suggests that deposit acceptance is fairly consistent across the days of the week, with a slight peak on Thursdays. Monthly trends show May as the strongest month for acceptances, while September and June are the weakest. The most striking insight is the overwhelming preference for 'cellular' contact for deposit acceptances, significantly dwarfing 'telephone' contact.


*  **Correlation Heatmap:**

    ![Correlation Heatmap](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/Heat_map_correlation.png)

**Key Observations from the Correlation Heatmap:**

*   **Strong Positive Correlations:**
    *   `emp.var.rate` and `euribor3m`: These variables show a very strong positive correlation, indicating that as the employment variation rate increases, the three-month Euro Interbank Offered Rate also tends to increase.
    *   `emp.var.rate` and `nr.employed`: A strong positive correlation suggests that higher employment variation rates are associated with a greater number of employees.
    *   `euribor3m` and `nr.employed`: This pair also exhibits a strong positive correlation, implying that as the Euro Interbank Offered Rate increases, the number of employees tends to increase as well.
*   **Moderate Negative Correlations:**
    *   `pdays` and `emp.var.rate`: There is a moderate negative correlation, suggesting that as the number of days since the client was last contacted from a previous campaign increases, the employment variation rate tends to decrease.
    *   `pdays` and `euribor3m`: A similar moderate negative correlation indicates that longer periods since the last contact are associated with lower Euro Interbank Offered Rates.
    *   `pdays` and `nr.employed`: This negative correlation suggests that as the number of days since the last contact increases, the number of employees tends to decrease.
*   **Other Notable Correlations:**
    *   `previous` and `poutcome`: These variables are correlated because `poutcome` (outcome of the previous marketing campaign) is directly related to whether the client was previously contacted (`previous`). If a client has been contacted before, the outcome of that contact will influence these variables.
*   **Low Correlations:**
    *   Features like `age`, `duration`, and `campaign` show relatively low correlations with the economic indicators (`emp.var.rate`, `euribor3m`, `nr.employed`). This suggests that these demographic and campaign-specific attributes do not have a strong linear relationship with the broader economic context.


## 3. Data Preparation

*   **Target Encoding:** The target variable `y` is renamed to `deposit` and encoded into numerical format: `{'no': 0, 'yes': 1}`.
*   **Feature Selection:** For initial modeling, only "bank client data" features are used: age, job, marital, education, default, housing, loan.
*   **Preprocessing Pipeline:**
    *   **Numerical Features (`age`):** Scaled using `StandardScaler`.
    *   **Categorical Features:** One-hot encoded using `OneHotEncoder` with `drop='first'` (to avoid multicollinearity) and `handle_unknown='ignore'`.
    *   A `ColumnTransformer` is used to apply these transformations to the respective column types.
*   **Train/Test Split:** The data is split into training and testing sets with a `test_size` of 20% and `stratify=y` to maintain the class distribution in both sets, crucial for imbalanced datasets.
    *   `X_train` shape: (32950, 28)
    *   `X_test` shape: (8238, 28)
    *   `y_train` value counts: 0 (29238), 1 (3712)
    *   `y_test` value counts: 0 (7310), 1 (928)

## 4. Modeling

The project compares the performance of several classification algorithms, starting with a baseline model and then moving to more complex models.

**Baseline Model: Dummy Classifier**

A `DummyClassifier` with a `stratified` strategy is used as a simple baseline. This classifier makes predictions randomly based on the training set's class distribution, serving as the minimum performance our actual models should exceed.


**Dummy Classifier Performance**

| Metric            | Value   |
| :---------------- | :------ |
| Fit Duration (s)  | 0.0056  |
| Test Accuracy     | 0.8037  |
| Train Accuracy    | 0.8002  |
| Accuracy          | 0.8037  |
| Recall (Class 1)  | 0.1196  |
| F1 Score (Class 1) | 0.1207  |
| ROC AUC           | 0.5051  |

**Confusion Matrix:**

|                 | Predicted Negative | Predicted Positive |
| :-------------- | :----------------- | :----------------- |
| Actual Negative | 6510               | 800                |
| Actual Positive | 817                | 111                |

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.89      0.89      0.89      7310
           1       0.12      0.12      0.12       928

    accuracy                           0.80      8238
   macro avg       0.51      0.51      0.51      8238
weighted avg       0.80      0.80      0.80      8238
```
 ![BaseModel-DummyClassifier](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/BaseModel-DummyClassifier.png)


*   **Key Performance Metrics (Dummy Classifier):**
    *   Accuracy: 0.8037 (80.37%)
    *   Recall (Class 1): 0.1196 (approx. 12%)
    *   F1 Score (Class 1): 0.1207
    *   ROC AUC: 0.5051 (close to random guessing)
*   **Prediction Pattern:** The model primarily predicts the majority class (0), reflecting the data's imbalance.

   ![BaseModel-DummyClassifier](https://github.com/soureddy81/AIML_Bank_Markting_PA_III/blob/main/images/BaseModel-DummyClassifier.png)


**Simple Model: Logistic Regression**

A `LogisticRegression` model is built and evaluated.

*   **Key Performance Metrics (Logistic Regression):**
    *   Accuracy: 0.8874 (88.74%)
    *   Recall (Class 1): 0.00
    *   F1 Score (Class 1): 0.00
    *   ROC AUC: 0.6492
*   **Prediction Pattern:** Similar to the dummy classifier, this model also largely fails to identify positive cases (class 1), always predicting the majority class (0) due to class imbalance. The high accuracy is misleading.

**Model Comparisons (KNN, Decision Tree, SVM)**

The notebook then proceeds to compare Logistic Regression with K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM) using their default settings. The goal is to evaluate their performance in terms of accuracy, recall, F1-score, ROC AUC, and fit duration.

The results from these models are then compiled into a comparison table to facilitate analysis.

**Expected Comparison Table Structure:**


| Model                                                                        | Accuracy | Recall   | F1 Score | ROC AUC | Fit Duration (s) | Train Accuracy | Test Accuracy |
| :--------------------------------------------------------------------------- | :------- | :------- | :------- | :------ | :--------------- | :------------- | :------------ |
| Logistic Regression                                                          | 0.887351 | 0.000000 | 0.000000 | 0.648984 | 0.026559         | 0.887344       | 0.887351      |
| KNN                                                                            | 0.879704 | 0.076509 | 0.125331 | 0.596824 | 0.006186         | 0.891745       | 0.879704      |
| Decision Tree                                                                | 0.863802 | 0.086207 | 0.124805 | 0.574773 | 0.054061         | 0.917056       | 0.863802      |
| SVM                                                                            | 0.887473 | 0.002155 | 0.004296 | 0.544483 | 60.062488        | 0.887557       | 0.887473      |
| DummyClassifier                                                              | 0.803714 | 0.119612 | 0.120718 | 0.505086 | 0.005601         | 0.800243       | 0.803714      |


**Analysis of Results:**

*   **Logistic Regression:**
    *   The Logistic Regression model shows a high accuracy (0.887351), but a very poor recall (0.000000) and F1 Score (0.000000). This indicates that while it correctly predicts the majority class (non-subscribers), it fails to identify any of the subscribers. The ROC AUC of 0.648984 suggests a slightly better-than-random ability to distinguish between classes, but this is not reflected in its predictive performance for the minority class.
*   **K-Nearest Neighbors (KNN):**
    *   KNN achieves a reasonable accuracy (0.879704), but its recall (0.076509) and F1 Score (0.125331) are still low, indicating limited success in identifying subscribers. The ROC AUC of 0.596824 is better than the Dummy Classifier but still not satisfactory.
*   **Decision Tree:**
    *   The Decision Tree model has an accuracy of 0.863802, with a slightly improved recall (0.086207) and F1 Score (0.124805) compared to KNN. However, the ROC AUC of 0.574773 suggests that the model's ability to discriminate between classes is limited. The high train accuracy (0.917056) indicates that the model may be overfitting the training data.
*   **Support Vector Machine (SVM):**
    *   SVM shows a high accuracy (0.887473), but a very low recall (0.002155) and F1 Score (0.004296), similar to Logistic Regression. The ROC AUC of 0.544483 indicates a poor ability to distinguish between classes. The extremely long fit duration (60.062488 seconds) is a significant drawback.
*   **Dummy Classifier:**
    *   As expected, the Dummy Classifier has the lowest accuracy (0.803714), recall (0.119612), and F1 Score (0.120718). The ROC AUC of 0.505086 is close to random guessing, confirming its role as a baseline model.

**Key Observations:**

1.  **Class Imbalance Issue:** All models struggle to identify the minority class (subscribers) due to the significant class imbalance in the dataset. The high accuracy scores are misleading because they primarily reflect the models' ability to predict the majority class.
2.  **Poor Recall and F1 Scores:** The low recall and F1 scores for all models indicate that they are not effectively capturing the patterns of the subscriber class. This is a critical issue that needs to be addressed.
3.  **Limited Discriminative Power:** The ROC AUC scores are generally low, suggesting that the models have limited ability to discriminate between subscribers and non-subscribers.
4.  **Overfitting:** The Decision Tree model shows signs of overfitting, as indicated by the high train accuracy and relatively lower test accuracy.
5.  **Computational Cost:** The SVM model has a very high computational cost, making it impractical for large datasets or real-time applications.

*(The actual values for KNN, Decision Tree, and SVM would be populated after running the notebook.)*


## 5. Evaluation

**Initial Model Shortcomings:** Both the `DummyClassifier` and the `LogisticRegression` models, despite showing high overall accuracy, fail miserably at identifying the minority class (subscribers). This is a direct consequence of the severe class imbalance.

**Metric Choice:** For imbalanced datasets, Accuracy is a poor metric. Recall, F1-Score, and ROC AUC are more appropriate to evaluate the model's ability to correctly identify positive instances.

**Addressing Imbalance:** The notebook should ideally explore techniques to address class imbalance (e.g., SMOTE for oversampling, adjusting class weights, or using different evaluation metrics/thresholds) to build more useful predictive models.  Param Grid and Grid Search and SMOTE are used in the improvement of model later in the notebook.


## 6. Addressing Class Imbalance and Model Optimization

To address the limitations of the initial models, the following steps were taken:

1.  **SMOTE (Synthetic Minority Oversampling Technique):** SMOTE was applied to oversample the minority class in the training data. This helps to balance the class distribution and allows the models to better learn the patterns of the minority class.

2.  **Parameter Grid and Grid Search:** `GridSearchCV` was used to tune the hyperparameters of the models. A parameter grid was defined for each model, and `GridSearchCV` systematically searched through all possible combinations of hyperparameters to find the best performing model.

The following models were evaluated using SMOTE and Grid Search:

*   Logistic Regression
*   K-Nearest Neighbors (KNN)
*   Decision Tree
*   Support Vector Machine (SVM)

**Model Performance Comparison:**

| Model                       | Accuracy   | Recall   | F1 Score | ROC AUC   | Fit Duration (s) | Train Accuracy | Test Accuracy |
| :-------------------------- | :--------- | :------- | :--------- | :-------- | :--------------- | :------------- | :------------ |
| DecisionTree+GridSearch     | 0.697014   | 0.470905 | 0.259347   | 0.636983  | 4.230701         | 0.698756       | 0.697014      |
| LogReg+SMOTE+GridSearch     | 0.584972   | 0.633621 | 0.255930   | 0.652689  | 10.727638        | 0.592534       | 0.584972      |
| SVM+SMOTE+GridSearch        | 0.499272   | 0.716595 | 0.243813   | 0.644063  | 7247.339279      | 0.509712       | 0.499272      |
| KNN+GridSearch              | 0.752853   | 0.330819 | 0.231698   | 0.597951  | 85.534759        | 0.795296       | 0.752853      |

**Analysis of Results:**

*   **Decision Tree with Grid Search:** Achieved an accuracy of 0.697014 and a recall of 0.470905. The F1 score is relatively low at 0.259347, indicating a trade-off between precision and recall.
*   **Logistic Regression with SMOTE and Grid Search:** Showed a lower accuracy of 0.584972 but a higher recall of 0.633621. This model is better at identifying positive cases but has a higher rate of false positives.
*   **SVM with SMOTE and Grid Search:** Had the lowest accuracy at 0.499272 but the highest recall of 0.716595. This model prioritizes identifying as many positive cases as possible, even at the cost of many false positives. The fit duration is extremely long, indicating high computational cost.
*   **KNN with Grid Search:** Achieved an accuracy of 0.752853 but a low recall of 0.330819. This model is more conservative in predicting positive cases, resulting in fewer false positives but also more false negatives.

**Observations:**

*   SMOTE helps to improve the recall of the models, particularly for Logistic Regression and SVM.
*   Grid Search optimizes the hyperparameters of the models, leading to improved performance compared to the initial models with default settings.
*   There is a trade-off between precision and recall. Models with higher recall tend to have lower precision, and vice versa.

## 7. Next Steps and Recommendations


**Future Improvements:**

*   **More Extensive Hyperparameter Tuning:**
    *   Expand the hyperparameter grids for each model to explore a wider range of parameter values.
    *   Use techniques like RandomizedSearchCV to efficiently search through a large hyperparameter space.
    *   Consider using Bayesian optimization for hyperparameter tuning, which can be more efficient than grid search or randomized search.
*   **Feature Engineering:**
    *   Create new features based on domain knowledge or insights from EDA.
    *   Explore different feature scaling techniques, such as MinMaxScaler or RobustScaler.
    *   Consider using feature selection techniques to reduce the dimensionality of the data and improve model performance.
*   **Ensemble Methods:**
    *   Experiment with different ensemble methods, such as Random Forests, Gradient Boosting Machines (GBM), and Stacking.
    *   Tune the hyperparameters of the ensemble methods to optimize their performance.
    *   Consider using a combination of different ensemble methods to create a more robust and accurate model.
*

*   **Cost-Benefit Analysis:**
    *   Conduct a cost-benefit analysis to determine the optimal balance between precision and recall.
    *   Consider the cost of false positives (e.g., wasted marketing efforts) and false negatives (e.g., missed opportunities).
    *   Use the results of the cost-benefit analysis to choose the model that maximizes the overall return on investment.

**Recommendations:**

Based on the analysis of the models and the potential future improvements, the following recommendations are made:

1.  **Prioritize Recall:** Given the business objective of identifying potential subscribers, prioritize models with higher recall, even if it means sacrificing some precision. SVM with SMOTE and Grid Search is a good starting point, but further optimization is needed to reduce the computational cost.

2.  **Invest in Feature Engineering:** Explore new features that could improve the model's ability to discriminate between subscribers and non-subscribers. This could include features related to customer demographics, financial history, or interaction with the bank.

3.  **Consider Ensemble Methods:** Ensemble methods, such as Random Forests or Gradient Boosting Machines, could potentially improve the model's performance by combining the strengths of multiple models.

4.  **Monitor and Retrain Models:** Continuously monitor the performance of the models and retrain them as needed to ensure that they remain accurate and effective. This is particularly important in dynamic environments where customer behavior and market conditions may change over time.

5.  **Further Data Collection:** Collect more data to improve the model's ability to generalize to new customers. This could include data from different marketing campaigns, different customer segments, or different time periods.

6.  **Business Context Integration:** Integrate the model into the bank's marketing processes and systems. This will allow the bank to use the model to target potential subscribers more effectively and improve the overall return on investment of its marketing campaigns.

7.  **A/B Testing:** Implement A/B testing to compare the performance of the model-driven marketing campaigns with traditional marketing campaigns. This will allow the bank to quantify the benefits of using the model and justify the investment in its development and maintenance.
```

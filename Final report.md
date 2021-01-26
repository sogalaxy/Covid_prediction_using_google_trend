# INTRODUCTION
  
Coronavirus disease(COVID-19) have been changed our life. 
The World Health Organization (WHO) has declared the COVID-19 outbreak as a global pandemic. 
The rapid increase in patient numbers has caused a shortage of hospital's capabilities. 
A patient who has been infected with COVID-19 showed some symptoms such as fever, fatigue, chills, headache, myalgia, arthralgia, nausea, swelling, diarrhea, redness, and cough. 
These symptoms are used to develop an mRNA vaccine for COVID-19([1 2]). 
All signs showed a different level depend on individual patients. 


In my project, google search trend data have been used as training data. 
This is because collecting a lot of patient's private clinical data is difficult. 
However, internet users' online research activity is recorded on the internet. 
So, it offers the potential for tracking public health and people's interest. 
Moreover, we can obtain the data from google research trends and observe the change in specific keywords. 
I assume that if people have COVID-19 related symptoms, they will google the sign and covid. 
So, I have collected the data from google trend using keywords individually('symptom + covid"). 
The google trend gives a time series ranging from 0 to 100, which corresponds to the query term's search intensity. 
Since Google Trend data is already processed by Google(7), It is not easy to predict the actual number of the patient. 

However, by simplify the data increase and decrease, we can reduce the complexity and expect the increase and decrease. 
By using the daily searching trend, I calculated the increase and decrease of the searching trend. 

If the google trend 'fever covid' value is 46 on April 18 and 34 on April 17, 
I set google trend values is increased, '1'.(Orange, TABLE1) and if it is decreased, I set it as '0'.


![1](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/1.jpg)

The number of COVID-19 Cases data is used from the CDC's daily trends cases. 
I judged the increase and decrease of the daily patient by comparing seven days after searching. 
For example, on April 16, the 'fever covid' search trend increased, '1', then we compare today's patient number increase and seven days after the patient number increase. 
If 7days after patients are growing, the predicted number of patients is increased('1') on April 17. (Yellow, TABLE1). 

![2](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/2.png)


![3](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/3.png)



# RESULT

In this project, six algorithms are calculated naïve--Bayes, Decision tree, Random forest, SVM, ANN, and KNN. 
Two hundred forty-five sets of dates have been used for training and 22 sets for the test. 
(20200301~20201030 for train and 20201101~20201123 for the test). Training data is randomly selected(220, 80%) from the 245 data set.  
All prediction is repeated 20 times and get mean Score and standard deviation.
To evaluate the algorithm, precision, recall, F1 Score, and accuracy is calculated from a confusion matrix. 
(Precision (positive predictive value) is how many are actually positive out of all the predicted positive classes(Precision = TP/ TP+FP). 
Recall (sensitive) is how are expected positive out of all positive class(Recall = TP / TP+FN). 
Accuracy is how many are predicted correctly out of all the classes (Accuracy = TP+TN /TP+FP+TN+FN). 
F1 Score is a measure of the test's accuracy considering recall and precision ( 2(Recall*Precision) / (Recall + Precision)

The first prediction is calculated by naïve--Bayes, Decision tree, and Random forest algorithm. 
the naïve-Bayes algorithm is a classification technique based on Bayes' theorem,P(├ c┤|x)=P(x│c)P(c)/P(y)  (P(├ c┤|x)  is posterior probability,P(x│c)  is likelihood,P(c)  is class prior probability,P(x)  is predctor prior probability). 
Naïve-Bayes assume all feature is independence between predictors. It is easy to build and understand. 
If each feature in the dataset has is binary type, we can use the Bernoulli Naïve Bayes classifier. 
The decision tree is the type of supervised learning algorithm that is used for the classification problem. We can make a distinct group by splitting data using a specific threshold value. 
The decision tree can classify the data and sort them down the tree. Therefore, it is easy to understand and compute fast. 
It's group's impurity can be calculated using the Gini index( 1 - ∑_(k=1)^k▒p_k^2 ) (pk = the proportion of class ). 
The random forest algorithm is similar to the decision tree, but it considers the randomly selected features in the dataset. 
To find the optimal number of the dataset, 100 to 2000 times of data were chosen randomly. 
By randomly assigning the element, Random forest can find the best-fit feature for classifying the results. 
The random forest algorithm takes more computing time than the decision tree because it considers all mixed features and the number of trees. 
In the Decision tree and random forest tree, we can find which feature most influences the results.

![4](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/4.png)


Random forest's Score shows the high scores in all scores. 
When we set the maximum decision tree's depth 3 to visualize, 'arthralgia covid' is the root node and decision nodes are 'fatigue covid' and 'joint pain covid.' 
Gini index is the lowest,0, when 'arthralgia covid,' 'joint pain' 'swelling covid' is increased. 
However, the number of samples is only 4, so it's hard to say this leaf node is significant. 
In the Random forest's tree, 'arthralgia covid' is also the root node and decision nodes are 'fatigue covid' and 'fever covid' as follows. 
Mean square error is calculated to evaluate each node. 'arthralgia covid' -> 'fever covid' tree showed lower than 'arthralgia covid' -> 'fatigue covid' (0.173 -> 0.22 ), but it also has less sample number

![5](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/5.png)


![6](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/6.png)



When we consider the feature importance of each tree, the decision tree showed 'joint pain covid' is the most important feature for classifying patient increasing and decreasing. 
However, in the random forest tree, the most important feature is 'fever covid,' and the overall feature importance values are reduced. 
As a result, I think the random forest is best fit model for prediction in the first prediction. 
Because it showed a high Score in prediction, recall, and accuracy, it also considers all the features in the data set.

![7](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/7.png)


![8](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/8.png)



In the second prediction, Support-Vector Machines(SVM) and Artificial Neural network(ANN) algorithm are used. 
Support-Vector Machines(SVM) is the supervised learning model used for classification and regression analysis. 
This algorithm classifies the different groups by drawing the line between two and calculating the maximal margin between them. 
Artificial Neural networks (ANN) have used a similar pattern of calculation as our brain. 
Each perceptron(single layer of neural network) are connected and become a new feature. 
Since we can use more input features by changing the network architecture, ANN can be powerful in predicting the output with limited inputs. 
The optimal number of layers is calculated by increasing the layer and comparing accuracy. Moreover, KNN is also used for classification and regression problems. 
KNN is one of the simple algorithms that classify the input by determining its similarity near value based on each data's distance.
The result showed that SVM significantly high in recall(sensitivity), and precision and accuracy is similar in both algorithms. 

![9](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/9.png)





# CONCLUSION

When we compared all algorithm's Scores, the Random forest algorithm showed the highest precision and accuracy, and SVM showed the highest in recall(sensitivity). 
In clinical prediction, recall might be more important than others because we need to prevent worst-case inpatient. 
As a result, I think the SVM algorithm would best fit my COVID-19 prediction project.

![10](https://github.com/sogalaxy/Covid_prediction_using_google_trend/blob/main/images/10.png){: .align-center}



All data have been used in this project from Google's preprocessed depends on the searching intensity. 
Therefore, the impact of each feature didn't consider. 
If we know each symptom's raw data, we might be able to estimate which symptom is more critical than others. 
Moreover, The symptom data do not come from one patient but are collected nationwide. 
So, this prediction might only use to check the general trend.
Furthermore, one big assumption of this project is that if people have any symptoms, they will search for google. 
However, the number of asymptomatic patients is increased. 
To increase the prediction, we need to consider how asymptomatic patients' behavior before testing COVID-19.


# Reference

1. L. A. Jackson et al., An mRNA Vaccine against SARS-CoV-2 — Preliminary Report. New England Journal of Medicine 383, 1920-1931 (2020).
2.	M. J. Mulligan et al., Phase I/II study of COVID-19 RNA vaccine BNT162b1 in adults. Nature 586, 589-593 (2020)
3. C. An et al., Machine learning prediction for mortality of patients diagnosed with COVID-19: a nationwide Korean cohort study. Scientific Reports 10, 18716 (2020).
4.	E. Allae, A. Mohamed, B. Abdessamad, B. Mouad, Research Square,  (2020).
5.	D. Assaf et al., Utilization of machine-learning models to accurately predict the risk for critical COVID-19. Internal and Emergency Medicine 15, 1435-1443 (2020).
6.	H. Yao et al., Severity Detection for the Coronavirus Disease 2019 (COVID-19) Patients Using a Machine Learning Model Based on the Blood and Urine Tests. Frontiers in Cell and Developmental Biology 8,  (2020).
7.	S. Ning, S. Yang, S. C. Kou, Accurate regional influenza epidemics tracking using Internet search data. Scientific Reports 9, 5238 (2019).

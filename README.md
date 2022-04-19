# lane-change-prediction
This is a project to predict the lane change intention and trajectory incorporating traffic context information
In this study, a lane change prediction model is proposed. Predicting lane change maneuvers is critical for autonomous vehicles and traffic management as lane change may cause conflict in traffic flow. This study aims to establish an integrated lane change prediction model incorporating traffic context using machine learning algorithms. In addition, lane change decisions and lane change trajectories are both predicted to capture the whole process. The framework of the proposed model contains two parts: the traffic context classification model, which is used to predict traffic level and vehicle type, and the integrated lane change prediction model, which is used to predict lane change decision with XGBoost and lane change trajectories with LSTM incorporating context information. Instead of considering lane change, we establish trajectory prediction models for left lane change and right lane change, further improving the prediction accuracy. 

 We can open the codes of traffic flow label, as well as the feature extraction. The xgboost-based prediction code and lstm-based trajectory prediction codes are all presented. 
 
Requirements

scipy>=0.19.0
numpy>=1.12.1
pandas>=0.19.2
sklearn
statsmodels
tensorflow>=1.3.0

The specific process of data preparation, traffic flow label, feature-extraction and prediction-model-building are carefully descripted in following steps:

1. Data Preparation
Researchers can apply for the highD dataset from the https://www.highd-dataset.com/. We preprocess the dataset to extract lane-keeping and lane-changeing samples (each sample is 10 s). The trajectory of samples includes the x-position, y-position, speed, acceleration, the gap and relative speed between subject vehicle and surrounding vehicles. 

2. Traffic flow label
The traffic flow is clustered into different levels according to the traffic density and traffic velocity in the paper. The density, velocity, and flow are calculated for each sample vehicle. Then the k-means method is applied to group the traffic into several levels.

3. Feature-extraction
The time-domain and frequency-domain features are both extracted from the trajectories as the inputs of the prediction model.

4. Lane change intention prediction
The xgboost algorithm is applied to predict the intention incorporating context information. The parameters of xgboost is tuned with grid-search method to obtain the optimal ones. 

5. Lane change trajectory prediction
The LSTM algorithm is applied to predict the lane change trajectory incorporating context information. 














# kaggle-tabular
## Purpose
This is to document the kaggle tabular series I have participated in.

## Some thoughts
1. EDA is likely the most critical part. See the EDA in may and the reference in the notebook. I have also tested letting the network learn the relation, but it perform poorer compared with the hand-engineered-feature version.

2. My experience with Optuna is not very positive, it seems in most cases, (random) grid search around a few parameters are good enough. In addition, I found the improvement from fine-tuning parameters is very limited for these datasets.  It might be more useful to blend/stack different models together within the given time frame. 

3. The work horse for the models so far is xgboost + fully connected nn, with occasionally gru, might worth trying adding more models. Haven't tried lightgbm, yet.

4. The apr_2022 is not solved yet. The median model is just slightly worse than some of the best models in the leader board.

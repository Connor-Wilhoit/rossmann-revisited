# rossmann-revisited
Besting publicly-available world-record exp_rmspe on the `Rossmann` Kaggle completion.

Reached a low (or high, depending upon how you look at it) Root Mean Squared Percentage Error (exp_rmspe) of < 0.10000.
Best score occured during epoch #4 of 6, and was ~0.098979.
Now the goal is to have a model consistently be under 0.10000, as the winner of the original Kaggle competition won with
a score (exp_rmspe) of 0.10021.

Note: You will need to either get the `rossmann` dataset and setup your directories prior to training, or simply modify
the first few lines of `rossmann_modeling.py`.  FastAI has built-in functions to download, unzip/untar, and setup correct directories if you so desire.

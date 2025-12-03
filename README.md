# ITEC 3040 Final Project

## install all packages needed to run the files by using the following line in the terminal
(python)/py -m pip install -r requirements.txt


<<<<<<< HEAD
## To the run GUI paste the following and run in the terminal 
(python)/py updated_GUI.py
Please use the downloadable template to see how to make a csv for batch data checks
=======
# To the run GUI paste the following and run in the terminal 
# (python)/py updated_GUI.py
# Please use the download template to see how to make a csv for batch data checks
>>>>>>> 3b8777fee7d9a87be7c6e7c0677cc6fff415bdee


This project uses the Bank Marketing dataset from a Portuguese bank’s direct phone campaigns (May 2008–Nov 2010). The goal is to predict whether a client will subscribe to a term deposit (y ∈ {yes, no}) based on client demographics, past contact history, and macroeconomic indicators. We use bank-additional-full.csv (N=41,188) for implementation, and evaluation will be done using a time-based split (train on earlier months, validate on later). The dataset is imbalanced, with only about 11.3% of clients subscribing (“yes”).

## Main Files
- README.md: Project overview and run instructions.
- requirements.txt: Python dependencies for the project.
- updated_GUI.py: GUI that loads pre-trained models from `Trained_models` (call/top-7/Q2) and supports single client predictions plus CSV batch runs.
- train_models.py: Offline trainer that reproduces the GUI’s training pipeline, saves `call_model.joblib`, `top7_model.joblib`, and `q2_model.joblib` into `Trained_models`.
- Question1.ipynb / Question2.ipynb: Notebook work for Q1 and Q2, including feature prep, model fitting, and evaluation for the call-count and high/low-group questions.
- Question3_reorganized.ipynb: Q3 feature selection and modeling notebook; defines the top-7 leak-free features used by the GUI/top7 model.
- initial_findings.ipynb: Early EDA and summary results; quick diagnostics on target balance, feature distributions, and baseline models.

## Features / Attributes 
* Client/Demographics: age, job, marital status, education, default (credit in default), housing (housing loan), loan (personal loan).
* Last Contact: contact type (cellular/telephone), month, day_of_week, duration (post-call; excluded from modelling due to leakage).
* Campaign History: campaign (contacts in current campaign), pdays (days since last contact, 999 = never), previous (contacts before current), poutcome (previous campaign outcome).
* Macroeconomic Context: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed.
* Target: y – whether the client subscribed to a term deposit.

## Initial Exploratory Findings
* Target distribution: 88.7% “no” vs. 11.3% “yes”
* Age trend: Older clients (66+) had the highest subscription rate (>40%).
* Job type: Students and retired clients show the highest success rates (~31% and 25% respectively), while blue-collar workers have the lowest (~7%).
* Education: Higher education levels correspond with higher likelihood to subscribe.
* Contact type: Cellular communication is far more effective (14.7% yes) than telephone (5.2% yes).
* Campaign performance: Success rate drops sharply after 3-4 contact attempts.



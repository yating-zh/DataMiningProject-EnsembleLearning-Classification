Readme
1. Coding environment
    - Operation system
      - The project was implemented on MacOS. The results can be reproduced on other operating systems which support python 3.
    - Programming language
      - This project was done in Jupyter Notebook, was converted to Python 3 script using this command:
      - %jupyter nbconvert --to script project_YatingZhang_s4797016.ipynb   
    - Packages installed
      - pandas
      - numpy
      - sklearn 
      - collections
    - Introduction
      - Firstly, the pre-processing technologies are applied (eg: dataset splitting, imputation, normalisation, one-hot encoding, etc). 
      - After the pre-processing phase, I implemented four types of classifiers: Decision Tree, Random Forest, k-NN, and Naive Bayes. Cross-validation is used for evaluation and hyperparameter tuning. Based on the evaluation of these classifiers, I chose the Random Forest, k-NN and Naive Bayes for ensembling. 
      - Finally, I did the same pre-processing steps for the test dataset and generated the prediction.
2. Instructions on how to run the codes
    - In the terminal, navigate to this code file directory, Run command:
    - %python3 project_YatingZhang_s4797016.py   
3. Reference
    - INFS7203 Learning resources - coding guides

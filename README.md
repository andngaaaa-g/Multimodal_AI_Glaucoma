# Multimodal_AI_Glaucoma
**Put this file in the code tab setting, it will make more sense that way**
a MultiModal_AI_Glaucoma model to predict whether the combination of parameter data and images will predict a status of glaucoma (GL), glaucoma suspect (GLS), or healthy eye (HC) status.
Both models in this repository use the same data mediums and also use the same supporting files. The difference in the two models codes are how the models are trained and how the data is relatviely split.

1st model instructions:
The first model is a train test split model that seperates the data into 80% training, 10% validation and 10% testing.
To run the first model's code, copy and paste these commands into terminal:
cd Desktop
cd F1
python glauc_mix_train.py --d image_dataset


2nd model instructions:
The second model is a 10 fold cross validation method, 8 folds for training, 1 fold for validation and 1 fold for testing.
each fold will be used as a valdiation fold within the inner fold.
each fold will be used as testing fold in the outer fold.
To run the second model's code, copy and paste these commands into terminal:
cd Desktop
cd F1
python nest_kfold_V6.py


Here is what the folder should look like when assembling the program to be used:
my-project/
├── README.md
├── image_dataset
├── requirements.txt
├── glauc_mix_train.py
├── glauc_mult_datasets.py
├── glauc_mult_models.py
├── nested_kfold_V6.py
├── __pycache__ (this folder will be automatically created after the model codes has been run the first time)



Don't reinvent the wheel and giving credit where it is due:
The code used in these programs are adapted from the code and study below:
#Credits @article{ahmed2016house, title={House price estimation from visual and textual features}, author={Ahmed, Eman and Moustafa, Mohamed}, journal={arXiv preprint arXiv:1609.08399}, year={2016} }
#Main Code that this was adapted from: https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/


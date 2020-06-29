# -*- coding: utf-8 -*-
"""
[He, Li]
[10006204]
[MMAI]
[2020]
[891]
[June 28, 2020]


Submission to Question [3], Part [1]
"""

# import important libraries
import re, unicodedata
import contractions
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from autocorrect import Speller
spell = Speller(lang='en')  # set spell checker's language to English
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from simpletransformers.classification import ClassificationModel

def main():
    # load train & test data
    df_train = pd.read_csv("sentiment_train.csv")
    df_test = pd.read_csv("sentiment_test.csv")

    #set random seed
    random = 42

    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(df_train['Sentence'], df_train['Polarity'], test_size=0.10, random_state=random)
    train_dataset =  pd.concat([X_train, y_train], axis=1)
    val_dataset = pd.concat([X_val, y_val], axis=1)

    # Load a pre-trained model, and train it with our data | See all models available: https://huggingface.co/transformers/pretrained_models.html
    # Create model ... args = parameters
    args={'reprocess_input_data': True, 'max_seq_length': 300, 'num_train_epochs': 1, 'fp16': False, 'train_batch_size': 4, 'overwrite_output_dir': True}
    my_model = ClassificationModel('roberta', 'distilroberta-base', num_labels=2, use_cuda=True, cuda_device=0, args=args)
    # Train the model
    my_model.train_model(train_dataset)

    # Evaluate the model
    result, model_outputs, wrong_predictions = my_model.eval_model(val_dataset, acc=f1_score)
    pred_val = np.argmax(model_outputs, axis=1).tolist()

    print("Results on evaluation:")
    print("----------------------")
    print("F1 Score = {:.6f}\n".format(f1_score(y_val, pred_val, average='micro') * 100))

    print(classification_report(y_val, pred_val))
    print(confusion_matrix(y_val, pred_val))

    # get results on test set
    pred_test, _ = my_model.predict(df_test['Sentence'])

    # print f1 score
    print(f1_score(df_test.Polarity, pred_test))

    # print accuracy score
    print(accuracy_score(df_test.Polarity, pred_test))

    # save input/ground truth/prediction as one csv
    df_test['prediction'] = pred_test
    df_test.to_csv('q3_ans.csv', index=False)

if __name__ == '__main__':
    main()
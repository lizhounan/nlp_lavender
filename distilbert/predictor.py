# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import warnings
warnings.simplefilter('ignore')
import os
import json
import pickle
import io
import sys
import signal
import traceback
import flask
import numpy as np
import pandas as pd
import re
import unicodedata
import contractions
import string
import time
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
from support import DistilBERTClass
import logging
logging.basicConfig(level=logging.ERROR)
def clean_text(text):
    # Lower casing
    text = text.lower()
    
    # Remove html codes
    text = re.sub(r"&amp;", " ", text)
    text = re.sub(r"&quot;", " ", text)
    text = re.sub(r"&#39;", " ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    
    # Strips (removes) whitespaces
    text = text.strip(' ')
    
    ################ Social media cleaning ############
    
    # Remove hashtags (Regex @[A-Za-z0-9]+ represents mentions and #[A-Za-z0-9]+ represents hashtags. )
    text = re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text)
    
    # Remove URLS (Regex \w+:\/\/\S+ matches all the URLs starting with http:// or https:// and replacing it with space.)
    text = re.sub("(\w+:\/\/\S+)", " ", text)
    text = re.sub(r'http\S+', ' ', text)
    
     # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove @users
    text = re.sub(r'@[\w]*', '', text)
    # remove Reddit channel reference /r
    text = re.sub(r'r/', '', text)
    
    # remove reddit username
    text = re.sub(r'u/[\w]*', '', text)
    # remove '&gt;' like notations
    text = re.sub('&\W*\w*\W*;', ' ', text)
    # remove hashtags
    text = re.sub(r'#[\w]*', '', text)
    ###################################################
    
    # Dealing with contractions
    text = contractions.fix(text)
    
    text = re.sub(r"what\'s", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can\'t", "can not ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"\'t", " not", text )
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"\'em'", " them ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    
    
    # Removes punctuations
    text = re.sub('['+string.punctuation+']', " ", text)
    
	# Removes non alphanumeric characters
    #text = re.sub('\W', ' ', text)
    
    # Removes non alphabetical characters
    text = re.sub('[^a-zA-Z]+', ' ', text)
    
    # Replaces all whitespaces by 1 whitespace
    text = re.sub('\s+', ' ', text)
    
    return text

        
device = 'cpu'
prefix='/opt/ml/'

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    tokenizer = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = DistilBERTClass()
            print('loading state dict')
            print('...')
            cls.model.load_state_dict(torch.load(prefix+'model/distilbert_demo_emotions_state_dict'), strict=False)
            print('loaded')
            cls.model.to(device)
            cls.model.eval()
            cls.model = torch.quantization.quantize_dynamic(cls.model, {torch.nn.Linear}, dtype=torch.qint8)
            cls.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
        return cls.model, cls.tokenizer
      
    @classmethod
    def predict(cls, texts):
        """For the input, do the predictions and return them."""
        MODEL, TOKENIZER = cls.get_model()
        pred = []
        for text in texts:
            input = TOKENIZER.encode_plus(
                        text,
                        None,
                        add_special_tokens=True,
                        max_length=100,
                        pad_to_max_length=True,
                        return_token_type_ids=True
                    )
            ids = torch.tensor([input['input_ids']], dtype=torch.long)
            mask = torch.tensor([input['attention_mask']], dtype=torch.long)
            token_type_ids = torch.tensor([input["token_type_ids"]], dtype=torch.long)
            # to device
            ids = ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            # predict
            output = MODEL(ids, mask, token_type_ids)
            pred.append(torch.sigmoid(output).detach().numpy()[0].tolist())
        return pred

# The flask app for serving predictions 
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    
    health, _ = ScoringService.get_model() 
    # health is not None  

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
        # data = np.squeeze(pd.read_csv(s, header=None).values)

    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))
    # preprocess
    data = data.astype(str)
    data.columns = ['text']
    data['text'] = data['text'].apply(clean_text)
    texts = data['text'].values

    # Do the prediction
    preds = ScoringService.predict(texts)
    preds = [[round(score, 3) for score in pred] for pred in preds ]

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame(preds, columns=['anger', 'fear', 'joy', 'sadness'], index=list(range(len(preds)))).to_csv(out)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')

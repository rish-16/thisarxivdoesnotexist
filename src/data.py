import pandas as pd
import numpy as np
import json
from simpletransformers.t5 import T5Model

PATH = "../arxiv-metadata-oai-snapshot.json"
N = 20

def get_metadata():
    with open(PATH, 'r') as f:
        for line in f:
            yield line
            
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    break
    
titles = []
ids = []
abstracts = []
authors_parsed = []
authors = []
years = []
metadata = get_metadata()

for paper in metadata:
    paper_dict = json.loads(paper)
    ref = paper_dict.get('journal-ref')
    try:
        year = int(ref[-4:]) 
        if 1992 < year < 2002:
            years.append(year)
            ids.append(paper_dict.get('id'))
            authors_parsed.append(paper_dict.get('authors_parsed'))
            authors.append(paper_dict.get('authors'))
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
    except:
        pass 

len(titles), len(abstracts), len(years)    

papers = pd.DataFrame({
    'id' : ids,
    'title': titles,
    'authors': authors,
    'authors parsed': authors_parsed,
    'abstract': abstracts,
    'year': years
})

# print(papers.head())

papers.to_csv('nlp.csv')

papers = papers[['title','abstract']]
papers.columns = ['target_text', 'input_text']
papers = papers.dropna()

eval_df = papers.sample(frac=0.2, random_state=101)
train_df = papers.drop(eval_df.index)

# print (train_df.shape, eval_df.shape)

train_df['prefix'] = "summarize"
eval_df['prefix'] = "summarize"

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create T5 Model
model = T5Model(model_type="t5", model_name="t5-base", args=model_args, use_cuda=True)

# Train T5 Model on new task
model.train_model(train_df)

# Evaluate T5 Model on new task
results = model.eval_model(eval_df)
print (results)

random_num = 350
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')
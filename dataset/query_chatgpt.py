import os
import json
from pathlib import Path
import openai
import timeout_decorator

openai.api = os.environ["OPENAI_API_KEY"]

dataset = 'asap-6'
prompt_version = '0405'

path = Path(f"./chatgpt_api_{prompt_version}/{dataset}/train.jsonl")
first_time= not path.is_file()
first_time = True
num = int(dataset.split('-')[1])
# load asap dataset
if first_time:
    import pandas as pd
    train_set = pd.read_csv("./asap-sas-splitted/original/train.tsv", sep='\t')
    test_set_label = pd.read_csv("./asap-sas-splitted/original/public_leaderboard_solution.csv")
    test_set = pd.read_csv("./asap-sas-splitted/original/public_leaderboard_rel_2.tsv", sep='\t')
    test_set['essay_score'] = test_set_label['essay_score']

    df_train = train_set[train_set['EssaySet'] == num]
    df_test = test_set[test_set['EssaySet']== num]
    
    train_jsonl = json.loads(df_train.to_json(orient='records'))
    test_jsonl = json.loads(df_test.to_json(orient='records'))
else:
    train_jsonl = []
    test_jsonl = []
    with open(f"./chatgpt_api_{prompt_version}/{dataset}/train.jsonl", 'r') as f:
        train_jsonl = [json.loads(each) for each in f.readlines()]
    with open(f"./chatgpt_api_{prompt_version}/{dataset}/test.jsonl", 'r') as f:
        test_jsonl = [json.loads(each) for each in f.readlines()]

prompt_path = f"chatgpt_api_{prompt_version}/prompts/{dataset}/prompt.txt"

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

def load_input(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
    return content

prompt_message = load_input(prompt_path)

def template(student_answer):
    return f"{prompt_message}\n\n[Student Answer]: {student_answer}\n[Score and Rationale]:"

# Set the timeout
@retry(wait=wait_random_exponential(min=30, max=80), stop=stop_after_attempt(2))
@timeout_decorator.timeout(40)
def chat_completion(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        n=1,
    )
    return completion.choices

from tqdm import tqdm

for line in tqdm(train_jsonl, total=len(train_jsonl), desc="Processing lines"):
    student_answer = line['EssayText']
    student_answer = student_answer.replace('^p',' ').replace('^P',' ')
    prompt = template(student_answer)
    if 'gpt-3.5-turbo' in line.keys() and len(line['gpt-3.5-turbo']) > 0:
        pass
    else:
        try:
            choices = chat_completion(prompt)
        except Exception as e: 
            print(e)
            choices = []
        outputs = []
        if len(choices) == 0:
            line['gpt-3.5-turbo'] = outputs
        else:
            for choice in choices:
                score_and_ratioanle = choice.message["content"]
                outputs.append({"index":choice.index ,"content": score_and_ratioanle})
            line['gpt-3.5-turbo'] = outputs

import jsonlines
with jsonlines.open(f"./chatgpt_api_{prompt_version}/{dataset}/train.jsonl", 'w') as writer:
    writer.write_all(train_jsonl)

for line in tqdm(test_jsonl, total=len(test_jsonl), desc="Processing lines"):
    student_answer = line['EssayText']
    student_answer = student_answer.replace('^p',' ').replace('^P',' ')
    prompt = template(student_answer)
    if 'gpt-3.5-turbo' in line.keys() and len(line['gpt-3.5-turbo']) > 0:
        pass
    else:
        try:
            choices = chat_completion(prompt)
        except Exception as e: 
            print(e)
            choices = []
        outputs = []
        if len(choices) == 0:
            line['gpt-3.5-turbo'] = outputs
        else:
            for choice in choices:
                score_and_ratioanle = choice.message["content"]
                outputs.append({"index":choice.index ,"content": score_and_ratioanle})
            line['gpt-3.5-turbo'] = outputs

with jsonlines.open(f"./chatgpt_api_{prompt_version}/{dataset}/test.jsonl", 'w') as writer:
    writer.write_all(test_jsonl)

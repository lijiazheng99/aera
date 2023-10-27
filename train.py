import aera
import typing
import argparse
from datasets import load_metric

import evaluate

from transformers import set_seed
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, LongT5ForConditionalGeneration
import aera
from aera.qwk import quadratic_weighted_kappa
import numpy as np

from datetime import datetime
now = datetime.now()
time = now.strftime("%m%d-%H%M")
from sklearn.metrics import accuracy_score,f1_score
accuracy = accuracy_score
f1_score = f1_score

import logging
import os
os.environ["WANDB_PROJECT"]= "student answer assessment"
# Set training log
logging.basicConfig(filename='./results/generation_training.log', level=logging.INFO)

import re
regex = r"\d+ point[s]?|No point"

def get_score(text):
    match = re.search(regex, text)
    if match:
        if match.group(0)[0] in ['0','1','2','3']:
            return int(match.group(0)[0])
        else:
            # score not in [0,1,2,3]
            return -1
    else:
        # No number matached
        return -1

def train_generation(output_name:str="", dataset:str="asap-1", random_seed:int=0, batch_size:typing.Optional[int]=8, num_train_epochs:typing.Optional[int]=30, report: typing.Optional[typing.List[str]] = None):

    train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args = aera.load_data_generation(dataset, random_seed, "google/long-t5-tglobal-large")

    model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-large")
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        pred_scores = [get_score(text) for text in decoded_preds]
        label_scores = [get_score(text[0]) for text in decoded_labels]
        qwk = quadratic_weighted_kappa(label_scores, pred_scores, min(label_scores), max(label_scores))

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        result['qwk'] = qwk

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    output_dir = aera.get_path(output_name)

    #T5 Typically, 1e-4 and 3e-4 work well for most problems
    logging_steps = len(train_dataset) // batch_size
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy = "epoch",
        learning_rate=1e-4,
        generation_max_length=180,
        predict_with_generate= True,
        metric_for_best_model = "eval_qwk",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_total_limit = 2,
        save_strategy = "epoch",
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_steps=logging_steps,
        report_to=report,
        run_name=output_name
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info(f'{output_name} seed:{random_seed} batch:{batch_size} epoches:{num_train_epochs}')
    print(output_name)
    trainer.train()
    trainer.model.save_pretrained(os.path.join(output_dir, 'checkpoint-best'))
    result = trainer.predict(test_dataset=test_dataset)
    decoded_preds = tokenizer.batch_decode(result.predictions, skip_special_tokens=True)

    df_test = dataset_args.df_test
    decoded_preds_labels = [int(get_score(t5_pred)) for t5_pred in decoded_preds]
    true_labels = [int(true_lab) for true_lab in df_test['gpt-3.5-turbo_top1_score']]
    acc = accuracy(y_pred=decoded_preds_labels, y_true=true_labels)
    f1 = f1_score(y_pred=decoded_preds_labels, y_true=true_labels, average='macro')
    qwk = quadratic_weighted_kappa(true_labels, decoded_preds_labels, min(true_labels), max(true_labels))
    
    df_test['t5_large_gen'] = decoded_preds
    df_test['t5_large_gen_label'] = decoded_preds_labels
    label_distri = ','.join([f'{index}:{value} ' for index, value in zip(df_test['t5_large_gen_label'].value_counts().index.tolist(), df_test['t5_large_gen_label'].value_counts().tolist())])
    logging.info(f'{output_name} acc:{"{:.2f}".format(acc*100)} f1:{"{:.2f}".format(f1*100)} qwk:{"{:.2f}".format(qwk*100)} bleu:{result.metrics["test_bleu"]} label distribution: {label_distri}')
    df_test.to_csv(f'./results/{output_name}.csv',index=None)
    return acc, f1, qwk, result.metrics["test_bleu"]

random_seeds = [210, 102, 231, 314, 146]

def main():
    parser = argparse.ArgumentParser(description="AERA explainable student answer assessment train")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset name. e.g., asap-1")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Batch size, default 8 ")
    parser.add_argument('-e', '--epoch', default=30, type=int, help="Epoch, default 30 ")
    parser.add_argument('-p', '--path', required=True, type=str, help="Output folder name ")
    parser.add_argument('-r', '--round', required=True, default=3, type=int, help="Rounds to train, default 3 ")

    args = parser.parse_args()
    random_seeds = [210, 102, 231, 314, 146]
    start = 0
    times_to_train = args.round
    batch_size = args.batch_size
    num_train_epochs = args.epoch
    dataset = args.dataset

    accs, f1s, qwks, bleus = [], [], [], []
    for idx in range(start, times_to_train):
        now = datetime.now()
        time = now.strftime("%m%d-%H%M")
        set_seed(random_seeds[idx])
        output_name = f'{args.path}-{time}-{idx}'

        acc, f1, qwk, bleu = train_generation(output_name=output_name, dataset=dataset, random_seed = random_seeds[idx], batch_size=batch_size, num_train_epochs=num_train_epochs, report="wandb")
        accs += [acc]
        f1s += [f1]
        qwks += [qwk]
        bleus += [bleu]
    logging.info(f'{args.path} ----- Average acc:{"{:.2f}".format(np.mean(accs)*100)} f1:{"{:.2f}".format(np.mean(f1s)*100)} qwk:{"{:.2f}".format(np.mean(qwks)*100)} bleu:{np.mean(bleus)}')

if __name__ == "__main__":
    main()
import aera

import os
import typing
import argparse
from datasets import load_metric

from transformers import set_seed
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

import torch
from torch.nn.functional import softmax
from aera.qwk import quadratic_weighted_kappa
import numpy as np
import logging

from datetime import datetime
now = datetime.now()
time = now.strftime("%m%d-%H%M")
accuracy = load_metric("accuracy", experiment_id = time)
f1_score = load_metric("f1", experiment_id = time)

logging.basicConfig(filename='./results/classification_training.log', level=logging.INFO)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    output = {}
    output.update(accuracy.compute(predictions=predictions, references=labels))
    output.update(f1_score.compute(predictions=predictions, references=labels, average='macro'))
    output.update({'qwk':quadratic_weighted_kappa(predictions, labels, min(labels), max(labels))})
    return output

def train(output_name:str="", model:str="", dataset:str="asap-1", random_seed:int=0, batch_size:typing.Optional[int]=8, num_train_epochs:typing.Optional[int]=30, report: typing.Optional[typing.List[str]] = None):

    train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args = aera.load_data(dataset, random_seed, model)

    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=dataset_args.num_labels)

    output_dir = aera.get_path(output_name)
    logging_steps = len(train_dataset) // batch_size
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy = "epoch",
        save_total_limit = 2,
        save_strategy = "epoch",
        metric_for_best_model = "eval_qwk",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end = True,
        weight_decay=0.01,
        logging_steps=logging_steps,
        report_to=report
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer)

    trainer.train()
    trainer.model.save_pretrained(os.path.join(output_dir, 'checkpoint-best'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'checkpoint-best'))
   
    result = trainer.predict(test_dataset=test_dataset)
    result_tensor = torch.from_numpy(result.predictions)
    result_ids = torch.argmax(softmax(result_tensor, dim=1), dim = 1).numpy()

    qwk = quadratic_weighted_kappa(result.label_ids, result_ids, min(result.label_ids), max(result.label_ids))
    dataset_args.df_test['pred'] = result_ids
    print(dataset_args.df_test['pred'].value_counts())
    dataset_args.df_test.to_csv(f'./results/{output_name}.csv',index=None)
    logging.info(f'{output_name}: Accuracy: {"{:.4f}".format(result.metrics["test_accuracy"])}, F1 Score: {"{:.4f}".format(result.metrics["test_f1"])}, QWK: {"{:.4f}".format(qwk)}')
    return result.metrics["test_accuracy"], result.metrics["test_f1"], qwk

def main():
    parser = argparse.ArgumentParser(description="AERA explainable student answer assessment train")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset name. e.g., asap-1")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Batch size, default 8 ")
    parser.add_argument('-e', '--epoch', default=30, type=int, help="Epoch, default 30 ")
    parser.add_argument('-p', '--path', required=True, type=str, help="Output folder name ")
    parser.add_argument('-m', '--model', required=True, type=str, help="Model name. e.g., bert-base-uncased")
    parser.add_argument('-r', '--round', required=True, default=3, type=int, help="Rounds to train, default 5 ")

    args = parser.parse_args()
    random_seeds = [210, 102, 231, 314, 146]
    start = 0
    times_to_train = args.round
    batch_size = args.batch_size
    num_train_epochs = args.epoch
    dataset = args.dataset
    model = args.model

    accs, f1s, qwks, bleus = [], [], [], []
    for idx in range(start, times_to_train):
        now = datetime.now()
        time = now.strftime("%m%d-%H%M")
        set_seed(random_seeds[idx])
        output_name = f'{args.path}-{time}-{idx}'

        acc, f1, qwk = train(output_name=output_name, model=model, dataset=dataset, random_seed = random_seeds[idx], batch_size=batch_size, num_train_epochs=num_train_epochs, report="wandb")
        accs += [acc]
        f1s += [f1]
        qwks += [qwk]
    logging.info(f'{args.path} ----- Average acc:{"{:.2f}".format(np.mean(accs)*100)} f1:{"{:.2f}".format(np.mean(f1s)*100)} qwk:{"{:.2f}".format(np.mean(qwks)*100)} ')

if __name__ == "__main__":
    main()

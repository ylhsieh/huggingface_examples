from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments,
                          EarlyStoppingCallback,
                         )
from datasets import (Dataset,
                      Features,
                      Value,
                      ClassLabel,
                      load_metric,
                     )
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
# ---------------------------------------------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# ---------------------------------------------------------------------------
categories = OrderedDict({
    1: 'class_1',
    2: 'class_2',
    0: 'others'
})
categories_to_id = OrderedDict([(type_name, type_id) for (type_id, type_name) in categories.items()])
max_length = 256
# ---------------------------------------------------------------------------
def tokenize_function(e):
    tokenized_batch = tokenizer(e['text'], padding="do_not_pad", truncation=True, max_length=max_length)
    return tokenized_batch

def padding_function(e, max_length):
    return_batch = tokenizer.pad(e, padding="max_length", max_length=max_length)
    if 'labels' in e:
        return_batch['labels'] = e['labels']
    return return_batch

metric = load_metric("accuracy")

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    
    return metric.compute(predictions=predictions, references=labels)
# ---------------------------------------------------------------------------
training_data = pd.read_csv('data-1.csv')
df = pd.DataFrame(training_data, columns=['text', 'labels'])
features = Features({'text': Value('string'),
                     'labels': ClassLabel(names=list(categories.keys()))})

ds_all = Dataset.from_pandas(df, features=features) \
                .map(tokenize_function, batched=True)

ds_all = ds_all.remove_columns(['text'])

for train_ids, test_ids in StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=14641).split(ds_all, ds_all['labels']):
    ds_train = Dataset.from_dict(ds_all[train_ids])
    ds_test = Dataset.from_dict(ds_all[test_ids])

ds_train = ds_train.map(padding_function, batched=True, fn_kwargs={"max_length": max_length})
ds_test = ds_test.map(padding_function, batched=True, fn_kwargs={"max_length": max_length})

ds_train.set_format('torch')
ds_test.set_format('torch')
# scale class weights
class_weights = [0.] * len(categories_to_id)
train_label_count = pd.DataFrame(ds_train['labels']).value_counts()
for i in range(len(categories_to_id)):
    class_weights[i] = round(1000. / train_label_count[i].item(), 4)
class_weights_pt = torch.tensor(class_weights)
class_weights_pt = class_weights_pt.to(device)
# ---------------------------------------------------------------------------
base_model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSequenceClassification.from_pretrained(base_model_name,
                                                           num_labels=len(categories.keys()), 
                                                           ignore_mismatched_sizes=True,
                                                           label2id=categories_to_id,
                                                           id2label=dict([((str(k), str(v))) for (k, v) in categories.items()]),
                                                          )
model = model.to(device)

loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_pt)

class SentTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        labels = inputs.get("labels")
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

num_epochs = 50

training_args = TrainingArguments(output_dir="hist_training",
                                  optim='adamw_torch',
                                  lr_scheduler_type='linear', 
                                  learning_rate=5e-5,
                                  per_device_train_batch_size=32,
                                  gradient_accumulation_steps=12,
                                  warmup_ratio=0.1,
                                  overwrite_output_dir=True, 
                                  num_train_epochs=num_epochs,
                                  evaluation_strategy="epoch", 
                                  save_strategy="epoch",
                                  logging_steps=10,
                                  load_best_model_at_end=True,
                                  save_total_limit=10,
                                 )

trainer = SentTrainer(model=model,
                      args=training_args,
                      train_dataset=ds_train,
                      eval_dataset=ds_test,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                     )

trainer_output = trainer.train()
# ---------------------------------------------------------------------------
model.save_pretrained('best_model') 
# ---------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained('best_model')
model = model.to(device)
model.eval()

predictions = []
answers = []

for batch in DataLoader(ds_test, batch_size=64):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    predictions.extend(list(prediction))
    answers.extend(batch["labels"])

report = classification_report(y_true=[categories[a.item()] for a in answers], 
                               y_pred=[categories[p.item()] for p in predictions],
                               output_dict=True,
                               )
print(report)

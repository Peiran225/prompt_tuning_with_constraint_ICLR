from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import Trainer, TrainingArguments
import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F


# Load the dataset (SST-5)
dataset = load_dataset("SetFit/sst5")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
tokenizer.pad_token_id = tokenizer.eos_token_id

# Tokenize the dataset
def preprocess_function(examples):
    inputs = ["Classify the sentiment: " + sentence for sentence in examples["text"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = examples["label"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets['validation'] = tokenized_datasets['test']
# Delete 'old_key'
del tokenized_datasets['test']

# Define custom metric function
def compute_metrics(eval_pred):
    # tokenizer = 
    import pdb;pdb.set_trace()
    label_text = ['terrible', 'bad', 'okay', 'good', 'great']
    # access_token = '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'
    # tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
    # tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left', token=access_token)
    # tokenizer = LlamaTokenizer.from_pretrained('chavinlo/alpaca-native', padding_side='left')
    # print("pad token id is none")
    # tokenizer.pad_token_id = tokenizer.eos_token_id
        
    pred_ids, labels = eval_pred
    total_predictions = len(labels)
    label2text = [label_text[labels[i].item()] for i in range(len(labels))] 
    pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # import pdb;pdb.set_trace()
    correct_predictions=0
    for labelt, predt in zip(label2text, pred_txt):
        print(predt)
        text = predt.lower()
        tmp=''
        verbalizer_dict = [k.lower() for k in label_text]
        for word in text.split():
            if word in verbalizer_dict:
                # corret_predictions += 1
                tmp = word
                break
        if labelt == tmp:
            correct_predictions += 1

        accuracy = correct_predictions / (total_predictions)
        return {"accuracy": accuracy, "eval_loss": 0.0}

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_dir='./logs',
)

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize the Trainer with the custom metric function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Pass the custom metrics function here
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

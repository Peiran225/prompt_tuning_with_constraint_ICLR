from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit, 
    TaskType,
)

from peft.tuners import PromptEmbedding
from datasets import load_dataset
import evaluate
import torch
import numpy as np
import argparse
import logging
import json
import os
import csv
from scipy.special import softmax
from my_trainer_current_final_ver import my_trainer
# from transformers import GPT2LMHeadModel,GPT2Config, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss

from data import load_prompt, CustomDataCollator 
from transformers import set_seed
# import wandb

PROMPT_DICT = {
        "NI": ['subtask047_misc_answering_science_questions', 'subtask034_winogrande_question_modification_object',
            'subtask028_drop_answer_generation', 'subtask054_multirc_write_correct_answer',
            'subtask019_mctaco_temporal_reasoning_category', 'subtask021_mctaco_grammatical_logical',
            'subtask027_drop_answer_type_generation', 'subtask038_qasc_combined_fact',
            'subtask029_winogrande_full_object', 'subtask033_winogrande_answer_generation',
            'subtask044_essential_terms_identifying_essential_words', 'subtask050_multirc_answerability',
            'subtask061_ropes_answer_generation', 'subtask002_quoref_answer_generation',
            'subtask037_qasc_generate_related_fact', 'subtask046_miscellaenous_question_typing',
            'subtask057_multirc_classify_incorrect_answer', 'subtask058_multirc_question_answering',
            'subtask006_mctaco_question_generation_transient_stationary',
            'subtask020_mctaco_span_based_question', 'subtask040_qasc_question_generation',
            'subtask042_qasc_incorrect_option_generation',
            'subtask008_mctaco_wrong_answer_generation_transient_stationary',
            'subtask023_cosmosqa_question_generation', 'subtask025_cosmosqa_incorrect_answer_generation',
            'subtask039_qasc_find_overlapping_words', 'subtask045_miscellaneous_sentence_paraphrasing',
            'subtask060_ropes_question_generation', 'subtask007_mctaco_answer_generation_transient_stationary',
            'subtask013_mctaco_answer_generation_absolute_timepoint', 'subtask059_ropes_story_generation',
            'subtask048_multirc_question_generation'],
        "PILE": ['prompt00', 'prompt01', 'prompt02', 'prompt03', 'prompt04', 'prompt05', 'prompt06', 'prompt07',
        'prompt08', 'prompt09', 'prompt10', 'prompt11', 'prompt12', 'prompt13', 'prompt14', 'prompt15',
        'prompt16', 'prompt17', 'prompt18', 'prompt19', 'prompt20', 'prompt21', 'prompt22', 'prompt23',
        'prompt24', 'prompt25', 'prompt26', 'prompt27', 'prompt28', 'prompt29'],
        "TRUE": {
            "SST-2": ["SST-2_0", "SST-2_1", "SST-2_2", "SST-2_3", "SST-2_4"],
            "sst-5": ["sst-5_0", "sst-5_1", "sst-5_2", "sst-5_3", "sst-5_4"],
            "agnews": ["agnews_0", "agnews_1", "agnews_2", "agnews_3", "agnews_4"],
            "trec": ["trec_0", "trec_1", "trec_2", "trec_3", "trec_4"],
            "subj": ["subj_0", "subj_1", "subj_2", "subj_3", "subj_4"]
        },
    }

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    
    
    set_seed(args.seed)

    model_name_or_path = args.path 
    
    # metric = evaluate.load("accuracy")
    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)

    # def compute_metrics(eval_pred):
    #     # import pdb;pdb.set_trace()
    #     predictions, labels = eval_pred
        
    #     # Detokenize predictions and labels
    #     # predicted_token_ids = np.argmax(predictions, axis=-1)
    #     predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    #     labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        
    #     # Tokenize into words for BLEU
    #     predictions = [pred.split() for pred in predictions]
    #     labels = [[label.split()] for label in labels]  # BLEU expects a list of references
        

    #     return metric.compute(predictions=predictions, references=labels)
    
    # def compute_metrics(eval_pred):
    #     import pdb;pdb.set_trace()
    #     logit, predictions, labels = eval_pred
        
    #     # Detokenize predictions and labels
    #     # predicted_token_ids = np.argmax(predictions, axis=-1)
    #     predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    #     labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        
    #     # Tokenize into words for BLEU
    #     predictions = [pred.split() for pred in predictions]
    #     labels = [[label.split()] for label in labels]  # BLEU expects a list of references
        

    #     return metric.compute(predictions=predictions, references=labels)

    if any(k in model_name_or_path for k in ("gpt2", "opt", "bloom", "llama", 'alpaca')):
        padding_side = "left"
    else:
        padding_side = "right"

    metric = evaluate.load("bleu")
    def load_model(model_name_or_path):
        if "llama" in model_name_or_path:
            model_name_or_path = "meta-llama/Llama-2-7b-hf"
            access_token = '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'
            tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left', token=access_token)
            model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16,token=access_token)
        elif 'alpaca' in model_name_or_path:
            special_tokens = {
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>"
                }

            tokenizer = LlamaTokenizer.from_pretrained('chavinlo/alpaca-native', padding_side='left')
            tokenizer.add_special_tokens(special_tokens)
            model = LlamaForCausalLM.from_pretrained('chavinlo/alpaca-native', torch_dtype=torch.float16)

            # model_name_or_path = 'chavinlo/alpaca-native'
        else:

            access_token = ''
            # tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, token=access_token)
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=access_token)
        if getattr(tokenizer, "pad_token_id") is None:
            print("pad token id is none")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer, model#, model_name_or_path

    def compute_metrics2(eval_pred):
        import pdb;pdb.set_trace()
        logits, labels = eval_pred
        batch_size = labels.shape[0]
        correct_predictions = 0
        total_predictions = len(labels)

        # 获取 "negative" 和 "positive" 的 token ID
        # negative_id = 4633#tokenizer.convert_tokens_to_ids("negative")
        # positive_id = 3967#tokenizer.convert_tokens_to_ids("positive")
        negative_id = 8178
        positive_id = 6374
        # negative_id = 19088
        # positive_id = 9432
        offset = 0
        # npdict = {negative_id:'Negative', positive_id}
        for i in range(batch_size):
            last_token_logit = logits[i, :]  # 取最后一个 token 的 logits
            last_token_label = labels[i, -1]  # 取最后一个 token 的标签
            if last_token_label == -100:
                offset += 1
                continue

            # 将logits转换为概率
            probs = softmax(last_token_logit)
            
            # 获取 "negative" 和 "positive" 的概率
            negative_prob = probs[negative_id].item()
            positive_prob = probs[positive_id].item()
            
            
            # 预测结果：哪个概率更大
            if negative_prob > positive_prob:
                predicted_label_id = negative_id
                print(f'Negative: negprob:{negative_prob} posprob:{positive_prob} label: {last_token_label.item()}')
            else:
                predicted_label_id = positive_id
                print(f'Positive: negprob:{negative_prob} posprob:{positive_prob} label: {last_token_label.item()}')
            
            # # 将预测的label转换为相应的ID
            # predicted_label_id = tokenizer.convert_tokens_to_ids(predicted_label)
            # true_label_id = tokenizer.convert_tokens_to_ids(tokenizer.decode(label))
            # true_label_id = labels[-1]

            # 比较预测的label和真实label
            if predicted_label_id == last_token_label.item():
                correct_predictions += 1

        accuracy = correct_predictions / (total_predictions-offset)
        return {"accuracy": accuracy, "eval_loss": 0.0}


    import re
    def compute_metrics(eval_pred):
        def extract_words(text):

            words = re.findall(r'\w+', text)
            return words
        # tokenizer = 
        # import pdb;pdb.set_trace()
        # label_text = ['terrible', 'bad', 'okay', 'good', 'great']
        label_text = ['negative', 'positive']
        access_token = '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'
        tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        # tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left', token=access_token)
        # tokenizer = LlamaTokenizer.from_pretrained('chavinlo/alpaca-native', padding_side='left')
        # print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
        pred_ids, labels = eval_pred
        total_predictions = len(labels)
        label2text = [label_text[labels[i].item()] for i in range(len(labels))] 
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # import pdb;pdb.set_trace()
        correct_predictions=0
        for labelt, predt in zip(label2text, pred_txt):
            print(predt + ' ==== ' + labelt)
            text = predt.lower()
            tmp=''
            verbalizer_dict = [k.lower() for k in label_text]
            for word in extract_words(text):
                if word in verbalizer_dict:
                    # corret_predictions += 1
                    tmp = word
                    break
            if labelt.lower() == tmp:
                correct_predictions += 1

        accuracy = correct_predictions / (total_predictions)
        return {"accuracy": accuracy, "eval_loss": 0.0}
    

    def compute_metrics_sst5(eval_pred):
        def extract_words(text):

            words = re.findall(r'\w+', text)
            return words
        # tokenizer = 
        # import pdb;pdb.set_trace()
        label_text = ['terrible', 'bad', 'okay', 'good', 'great']
        # label_text = ['negative', 'positive']
        access_token = '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'
        # tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left', token=access_token)
        # tokenizer = LlamaTokenizer.from_pretrained('chavinlo/alpaca-native', padding_side='left')
        # print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
        pred_ids, labels = eval_pred
        total_predictions = len(labels)
        label2text = [label_text[labels[i].item()] for i in range(len(labels))] 
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # import pdb;pdb.set_trace()
        correct_predictions=0
        for labelt, predt in zip(label2text, pred_txt):
            print(predt + ' ==== ' + labelt)
            text = predt.lower()
            tmp=''
            verbalizer_dict = [k.lower() for k in label_text]
            for word in extract_words(text):
                if word in verbalizer_dict:
                    # corret_predictions += 1
                    tmp = word
                    break
            if labelt.lower() == tmp:
                correct_predictions += 1

        accuracy = correct_predictions / (total_predictions)
        return {"accuracy": accuracy, "eval_loss": 0.0}


    def compute_metrics_subj(eval_pred):
        # tokenizer = 
        def extract_words(text):

            words = re.findall(r'\w+', text)
            return words
        # import pdb;pdb.set_trace()
        # label_text = ['terrible', 'bad', 'okay', 'good', 'great']
        label_text = ['objective', 'subjective']
        access_token = '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'
        tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        # tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left', token=access_token)
        # tokenizer = LlamaTokenizer.from_pretrained('chavinlo/alpaca-native', padding_side='left')
        # print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
        pred_ids, labels = eval_pred
        total_predictions = len(labels)
        label2text = [label_text[labels[i].item()] for i in range(len(labels))] 
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # import pdb;pdb.set_trace()
        correct_predictions=0
        for labelt, predt in zip(label2text, pred_txt):
            print(' '.join(extract_words(predt)) + ' ==== ' + labelt)
            text = predt.lower()
            tmp=''
            verbalizer_dict = [k.lower() for k in label_text]
            for word in extract_words(text):
                if word in verbalizer_dict:
                    # corret_predictions += 1
                    tmp = word
                    break
            if labelt.lower() == tmp:
                correct_predictions += 1

        accuracy = correct_predictions / (total_predictions)
        return {"accuracy": accuracy, "eval_loss": 0.0}
    
    def compute_metrics_trec(eval_pred):
        # tokenizer = 
        def extract_words(text):

            words = re.findall(r'\w+', text)
            return words
        # import pdb;pdb.set_trace()
        # label_text = ['terrible', 'bad', 'okay', 'good', 'great']
        label_text = ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number']
        access_token = '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'
        tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
        # tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, padding_side='left', token=access_token)
        # tokenizer = LlamaTokenizer.from_pretrained('chavinlo/alpaca-native', padding_side='left')
        # print("pad token id is none")
        tokenizer.pad_token_id = tokenizer.eos_token_id
            
        pred_ids, labels = eval_pred
        total_predictions = len(labels)
        label2text = [label_text[labels[i].item()] for i in range(len(labels))] 
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # import pdb;pdb.set_trace()
        correct_predictions=0
        for labelt, predt in zip(label2text, pred_txt):
            print(' '.join(extract_words(predt)) + ' ==== ' + labelt)
            text = predt.lower()
            tmp=''
            verbalizer_dict = [k.lower() for k in label_text]
            for word in extract_words(text):
                if word in verbalizer_dict:
                    # corret_predictions += 1
                    tmp = word
                    break
            if labelt.lower() == tmp:
                correct_predictions += 1

        accuracy = correct_predictions / (total_predictions)
        return {"accuracy": accuracy, "eval_loss": 0.0}

    tokenizer, model = load_model(model_name_or_path)
    if args.task=="SST-2":
        dataset = load_dataset("sst2")
        # Change1: Label text tokenizer function
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            label_text = ['negative', 'positive']
            # new_labels = [label_text[label] for label in examples['label']]
            # new_sentences = ["Sentence: " + sentence + "/Sentiment: " + new_labels[i] for i, sentence in enumerate(examples['sentence'])]
            template = "### Input:\n{input}\n\n### Response:\n{output}"
            new_labels = [label_text[label] for label in examples['label']]
            new_sentences = [template.format(input=sentence, output=new_labels[i]) for i, sentence in enumerate(examples['sentence'])]
            outputs = tokenizer(new_sentences, padding=True, truncation=True)
            # labels = tokenizer()
            # print(outputs.keys())
            outputs['new_sent'] = new_sentences
            # outputs['label_ids'] = outputs['input_ids'][:]
            # outputs['labels'] = [label for label in examples['label']]
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence"],
            )
        n_last_tokens = 1
        # tokenized_datasets['train']=tokenized_datasets['train'][:10]
        # tokenized_datasets['train'] = tokenized_datasets['train'].select(range(500))

    elif args.task=="sst-5":
        # import pdb;pdb.set_trace()
        dataset = load_dataset("SetFit/sst5")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            label_text = ['terrible', 'bad', 'okay', 'good', 'great']
            # outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            template = "### Input:\n{input}\n\n### Response:\n{output}"
            new_labels = [label_text[label] for label in examples['label']]
            new_sentences = [template.format(input=sentence, output=new_labels[i]) for i, sentence in enumerate(examples['text'])]
            outputs = tokenizer(new_sentences, padding=True, truncation=True) #, max_length=None)
            # outputs['label_ids']=outputs['input_ids'][:]
            outputs['new_text'] = new_sentences
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "label_text"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 5
    elif args.task=="agnews":
        dataset = load_dataset("ag_news")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 4
    elif args.task=="trec":
        dataset = load_dataset("trec")
        def rename_column(example):
            example['label'] = example['coarse_label']
            del example['coarse_label']
            return example
        dataset = dataset.map(rename_column)
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            label_text = ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number']
            template = "### Input:\n{input}\n\n### Response:\n{output}"
            new_labels = [label_text[label] for label in examples['label']]
            new_sentences = [template.format(input=sentence, output=new_labels[i]) for i, sentence in enumerate(examples['text'])]
            outputs = tokenizer(new_sentences, padding=True, truncation=True) #, max_length=None)
            # outputs['label_ids']=outputs['input_ids'][:]
            outputs['new_text'] = new_sentences
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "fine_label"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 6
    elif args.task=="subj":
        dataset = load_dataset("SetFit/subj")
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            # new_labels = [label_text for label in examples['label']]
            # label_text = ['subjective', 'objective']
            # outputs = tokenizer(examples["text"], padding=True, truncation=True) #, max_length=None)
            template = "### Input:\n{input}\n\n### Response:\n{output}"
            new_labels = [label for label in examples['label_text']]
            new_sentences = [template.format(input=sentence, output=new_labels[i]) for i, sentence in enumerate(examples['text'])]
            outputs = tokenizer(new_sentences, padding=True, truncation=True)
            # labels = tokenizer()
            # print(outputs.keys())
            outputs['new_sent'] = new_sentences
            # outputs['label_ids'] = outputs['input_ids'][:]
            return outputs
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "label_text"],
            )
        tokenized_datasets['validation'] = tokenized_datasets['test']
        # Delete 'old_key'
        del tokenized_datasets['test']
        num_label = 2

    

    print("finishing tokeninzing")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    import pdb; pdb.set_trace()
    data_collator = CustomDataCollator(tokenizer=tokenizer, padding="longest")
    
    # Train (original batches 32)
    training_args = TrainingArguments(
        output_dir=model_name_or_path + "-peft-prompt-tuning",
        learning_rate=args.learning_rate, #0.1 has great difference when using LM similarity, but the accuracy with layer -1 is low. 1e-3 if learning rate<=0.01, the projection of soft prompt will always be the original one
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epoch,
        weight_decay=0.01, #0.01
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # load_best_model_at_end=True,
        eval_accumulation_steps=2,
        seed=args.seed,
        # local_rank=-1
        # device='cuda:0'
    )

    training_args_LM = TrainingArguments(
        output_dir=model_name_or_path + "-peft-prompt-tuning2",
        learning_rate=args.learning_rate_LM, #0.1 has great difference when using LM similarity, but the accuracy with layer -1 is low. 1e-3 if learning rate<=0.01, the projection of soft prompt will always be the original one
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epoch,
        weight_decay=0.01, #0.01
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # load_best_model_at_end=True,
        eval_accumulation_steps=2,
        seed=args.seed,
        # local_rank=-1
        # device='cuda:0'
    
    )
    # add a hook to track the embeddings of middle layers of model.base_model
    def hook_fn(module, input, output):
        # import pdb;pdb.set_trace()
        # print("Hook function called with args:", input, "and kwargs:", output)
        # module.embedding_output = output
        setattr(module, 'embedding_output', output)

    # init_text = "What is the sentiment of this sentence? \n Positive , Negative."#"6.00 credit(s) to open a letter from her"
    if args.num_of_initial_text==1:
        prompt_names = [args.prompt]
    else:
        args.prompt_groups = ["TRUE", "NI", "PILE"]
        prompt_names = []
        for prompt_group in args.prompt_groups:
            if prompt_group == "TRUE":
                prompt_names += PROMPT_DICT[prompt_group][args.task]
            else:
                prompt_names += PROMPT_DICT[prompt_group]
    
    init_texts = [(prompt_name, load_prompt(args.prompts_dir, prompt_name, int(args.pile_len))) for prompt_name in
                   prompt_names]
    # import pdb;pdb.set_trace()



    
    if args.base_initial=="Random":
        post_dir = '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed) + '-random_init_baseline'
    elif args.particular_layer is not None:
        post_dir = '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed) + 'similarity' + str(args.similarity) + '-layer-' + str(args.particular_layer)
    else:
        post_dir = '-gamma-' + str(args.gamma) + '-lr-' + str(args.learning_rate) + '-lr_LM-' + str(args.learning_rate_LM) + '-epoch-' + str(args.epoch) + '-num_of_init_text-' + str(args.num_of_initial_text) + '-seed-' + str(args.seed) + 'similarity' + str(args.similarity)

    results_dir = 'results/' + model_name_or_path + '/' + args.prompt + '__' + args.task + post_dir + '.csv'
    directory = os.path.dirname(results_dir)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    new_file = True

    peft_config_without_layer = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=10,
        tokenizer_name_or_path=model_name_or_path,
        )
    for init_text_tuple, prompt_name in zip(init_texts, prompt_names): # range(len(init_texts)):
        init_text = init_text_tuple[1]
        

        #for initext in init_texts:
        org_input = tokenizer(init_text
                            , return_tensors='pt')
        num_virtual_tokens = len(org_input['input_ids'][0])
        
        peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=init_text,
        tokenizer_name_or_path=model_name_or_path,
        )

        
        
        #train with hook on different layer

        if model_name_or_path in ("gpt2", "bert-base-uncased"):
            each_layer = list(range(0,12))
        elif model_name_or_path=="gpt2-medium":
            each_layer = list(range(0,24))
        elif model_name_or_path=="gpt2-large":
            each_layer = list(range(0,36))
        elif model_name_or_path=="llama":
            each_layer = list(range(0,32))
        elif model_name_or_path=="FacebookAI/roberta-base":
            each_layer = list(range(0,12))
        elif model_name_or_path=="facebook/opt-125m":
            each_layer = list(range(0,32))
        elif 'llama' in model_name_or_path or 'Llama' in model_name_or_path or 'alpaca' in model_name_or_path:
            each_layer = list(range(32))


        a0 = -2
        a = [-2] + [-1] + each_layer
        final_acc_per_prompt = []
        print('baseline_only')
        print(args.baseline_only)
        if args.baseline_only==True:
            aa = [-1]
        elif args.particular_layer >-3:
            aa = [args.particular_layer]
        else:
            aa = a
        # import pdb;pdb.set_trace()
        for i in aa:
            if i == -2:
                # continue
                print("train the model when the hook layer is %s"% i)
                tokenizer, model = load_model(model_name_or_path)

                g_p = init_text #"What is the sentiment of this sentence? \n Positive , Negative."
                tokenized_g_p = tokenizer(g_p, padding="max_length", truncation=True, max_length=num_virtual_tokens)
                tokenized_g_p['input_ids'] = torch.tensor(tokenized_g_p['input_ids'])

                # print("training with penalized model")
                # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, cache_dir='/fs/nexus-scratch/peiran/.cache', num_labels=num_label)
                # model = GPT2LMHeadModel.from_pretrained('gpt2')
                # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=access_token)
                
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                    
                if any(k in model_name_or_path for k in ("gpt", "bert", "llama", 'apacha')):
                    model.config.pad_token_id = tokenizer.pad_token_id

                    
                if args.task == 'subj':
                    compute_metrics = compute_metrics_subj
                elif args.task == 'sst-5':
                    compute_metrics = compute_metrics_sst5
                elif args.task == 'trec':
                    compute_metrics = compute_metrics_trec

                trainer = my_trainer(
                    model=model,
                    args=training_args_LM,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    model_name_or_path=model_name_or_path,
                    tokenized_g_p=tokenized_g_p,
                    hook_layer=i,
                    similarity="L2",
                    gamma=args.gamma,
                    # tokenizer=tokenizer,
                ) 

                trainer.train()
                print("trainer.metrics_log %s"% trainer.metrics_log)
                final_acc_per_prompt.append(trainer.metrics_log[-1]['eval_accuracy'])
                outputs_list = trainer.metrics_log
                outputs_list = [{**d, 'layer': i, 'prompt': prompt_name} for d in outputs_list]
                if new_file == True:
                    headers = outputs_list[0].keys()
                    with open(results_dir, 'w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=headers)
                        # Write the headers (column names)
                        writer.writeheader()
                        for d in outputs_list:
                            writer.writerow(d)
                    new_file = False
                    print("new file %s"% new_file)
                 
                else:
                    print(new_file)
                    with open(results_dir, 'a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=headers)
                        for d in outputs_list:
                            writer.writerow(d)
            elif i == -1:
                # continue
                print("train the model when the hook layer is %s"% i)
                # wandb.init(project='prompting' + args.task, 
                #            entity='pyu123',
                #             config={ "model": args.path,
                #                         "learning_rate": args.learning_rate,
                #                         "learning_rate_LM": args.learning_rate_LM,
                #                         "gamma": args.gamma,
                #                         "epochs": args.epoch,
                #                         "layer": i
                #                         })
                # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=num_label)
                # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                tokenizer, model = load_model(model_name_or_path)
                if args.base_initial=="Random":
                    model = get_peft_model(model, peft_config_without_layer)
                else:
                    model = get_peft_model(model, peft_config)
                
                

                if any(k in model_name_or_path for k in ("gpt",)):
                    # print(model_name_or_path)
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "bert-base-uncased":
                    model.base_model.bert.encoder.layer[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "EleutherAI/gpt-j-6b":
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "FacebookAI/roberta-base":
                    model.base_model.base_model.encoder.layer[i].register_forward_hook(hook_fn) 
                elif any(k in model_name_or_path for k in ('llama', 'alpaca', 'Llama')):
                    model.base_model.base_model.layers[i].register_forward_hook(hook_fn)

                    
                
                if any(k in model_name_or_path for k in ("gpt", "bert", "llama")):
                    model.config.pad_token_id = tokenizer.pad_token_id
                

                trainer = my_trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics
                    # gamma=1e-4 #1e-4 will let aux_loss be in 1/10 of loss (around 0.8) at the beginning.
                ) 
                # eval = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
                # "Accuracy of projected soft prompt before train\n %s"% eval)


                trainer.train()
                print("trainer.metrics_log %s"% trainer.metrics_log)
                outputs_list = trainer.metrics_log
                outputs_list = [{**d, 'layer': i, 'prompt': prompt_name} for d in outputs_list]
                with open(results_dir, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    for d in outputs_list:
                        writer.writerow(d)
            else:
                print("train the model when the hook layer is %s"% i)
                # wandb.init(project='prompting' + args.task, 
                #            entity='pyu123',
                #             config={ "model": args.path,
                #                         "learning_rate": args.learning_rate,
                #                         "learning_rate_LM": args.learning_rate_LM,
                #                         "gamma": args.gamma,
                #                         "epochs": args.epoch,
                #                         "layer": i
                #                         })
                # compute the hook of the good_prompt
                tokenizer, model = load_model(model_name_or_path)
                g_p = init_text #"What is the sentiment of this sentence? \n Positive , Negative."
                tokenized_g_p = tokenizer(g_p, padding="max_length", truncation=True, max_length=num_virtual_tokens)
                tokenized_g_p['input_ids'] = torch.tensor(tokenized_g_p['input_ids'])

                # print("training with penalized model")
                # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, cache_dir='/fs/nexus-scratch/peiran/.cache', num_labels=num_label)
                # model = GPT2LMHeadModel.from_pretrained('gpt2')

                # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=access_token)
                model = get_peft_model(model, peft_config)
                # model.print_trainable_parameters()

                # add the hook to the ith layer of the model
                # import pdb;pdb.set_trace()
                if any(k in model_name_or_path for k in ("gpt",'gpt2')):
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "bert-base-uncased":
                    model.base_model.bert.encoder.layer[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "EleutherAI/gpt-j-6b":
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                elif model_name_or_path == "FacebookAI/roberta-base":
                    model.base_model.base_model.encoder.layer[i].register_forward_hook(hook_fn) 
                elif 'llama' or 'Llama' or 'alpaca' in model_name_or_path:
                    model.base_model.base_model.layers[i].register_forward_hook(hook_fn)
                elif 'gpt' in model_name_or_path:
                    model.base_model.transformer.h[i].register_forward_hook(hook_fn)
                
                    
                if any(k in model_name_or_path for k in ("gpt", "bert", "llama", 'alpaca')):
                    model.config.pad_token_id = tokenizer.pad_token_id
        
                trainer = my_trainer(
                    model=model,
                    args=training_args_LM,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    model_name_or_path=model_name_or_path,
                    tokenized_g_p=tokenized_g_p,
                    hook_layer=i,
                    similarity="L2_LM",
                    gamma=args.gamma
                ) 
                    
                trainer.train()
                print("trainer.metrics_log %s"% trainer.metrics_log)
                outputs_list = trainer.metrics_log
                outputs_list = [{**d, 'layer': i, 'prompt': prompt_name} for d in outputs_list]
                with open(results_dir, 'a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    for d in outputs_list:
                        writer.writerow(d)
                    
        if args.num_of_initial_text == 1:
            break
                
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=str, default = None)
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--path", default="FacebookAI/roberta-base", type=str)
    parser.add_argument("--hook_layer", default=-1, type=int)
    parser.add_argument("--prompts_dir", default="prompts/", type=str)
    parser.add_argument("--prompt_groups", default=["TRUE", ], type=list)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--task", default="trec", type=str)
    # parser.add_argument("--dataset", default="trec", type=str)
    parser.add_argument("--pile_len", default=-1, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--learning_rate_LM", default=0.01, type=float)
    parser.add_argument("--gamma", default=1e-5, type=float)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_of_initial_text", default=None, type=int)
    parser.add_argument("--particular_layer", default=-3, type=int)
    parser.add_argument("--baseline_only", default=False, type=bool)
    parser.add_argument("--base_initial", default="Text", type=str)

    args = parser.parse_args()
    
    # args.learning_rate = 1e-3
    # args.learning_rate_LM = 1e-3
    # args.gamma = 1e-8 # 1e-8 upgrade to 0.86 at layer 3 for gpt2 small, 5e-5 for bert
    # args.epoch = 1
    # args.path = "gpt2" #bert-base-uncased
    print(args)

     #gpt2=gpt small
    main(args)






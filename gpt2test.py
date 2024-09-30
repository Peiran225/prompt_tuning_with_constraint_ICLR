import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch.nn.functional as F

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda:1')
model.eval()  # Set the model to evaluation mode

# Load SST-5 dataset using Hugging Face datasets library
dataset = load_dataset("SetFit/sst5")  # SST-5 dataset is part of the 'sst' dataset

# Prepare the validation set
val_set = dataset['test']

# Define the SST-5 classes
label_text = ['terrible', 'bad', 'okay', 'good', 'great']

# Define a function to create a GPT-2 prompt for classification
def create_prompt(sentence):
    prompt = f"Classify the sentiment of the following sentence:\n\n\"{sentence}\"\n\nThe sentiment is: "
    return prompt

# Function to classify sentiment using GPT-2 with zero-shot classification
def classify_sentiment_gpt2(sentence):
    prompt = create_prompt(sentence)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda:1')
    
    # Generate output
    outputs = model.generate(**inputs, max_length=len(inputs["input_ids"][0]) + 20, num_return_sequences=1)
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated sentiment from the output
    for label in label_text:
        if label in generated_text.lower():
            return label
    
    return "unknown"

# Evaluate on validation set
correct = 0
total = len(val_set)
from tqdm import tqdm
for i, example in enumerate(tqdm(val_set)):
    # import pdb; pdb.set_trace()
    sentence = example['text']
    true_label = label_text[example['label']]  # Get the true label

    # Get the predicted label
    predicted_label = classify_sentiment_gpt2(sentence)
    
    print(f"Sentence: {sentence}")
    print(f"Predicted: {predicted_label}, True: {true_label}")
    
    # Compare the prediction with the true label
    if predicted_label == true_label:
        correct += 1

# Calculate accuracy
accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")


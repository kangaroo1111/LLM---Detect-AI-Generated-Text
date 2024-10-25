#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install transformers')


# In[3]:


get_ipython().system('pip install torch')


# In[12]:


get_ipython().system('pip install scikit-learn')


# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# In[7]:


#load in all the data sets

# kaggle_train = pd.read_csv('datasets/kaggle-data/train_essays.csv') 
# kaggle_prompts = pd.read_csv('datasets/kaggle-data/train_prompts.csv')

#The DAIGT_concatenated dataset can be downloaded from: https://www.kaggle.com/datasets/dsluciano/daigt-one-place-all-data?select=concatenated.csv
daigt_external_data = pd.read_csv('DAIGT_concatenated.csv')

# test_data_fixed = pd.read_csv('datasets/external-data/test_preprocessed_fixed.csv')
# # train_essays_v1 = pd.read_csv('datasets/external-data/train_essays_RDizzl3_seven_v1.csv')
# # train_data_fixed = pd.read_csv('datasets/external-data/train_preprocessed_fixed.csv')
# # train_v2_raw = pd.read_csv('datasets/external-data/train_v2_drcat_02_raw.csv')


# In[8]:


# Count 1s and 0s in 'generated' column for daigt_external_data
daigt_external_counts = daigt_external_data['generated'].value_counts()

print("\nDAIGT External Data - 'generated' column counts:")
print(daigt_external_counts)


# Based on the above output (Kaggle Train - 'generated' column counts:
# 0    1375
# 1       3
# DAIGT External Data - 'generated' column counts:
# 0    29907
# 1    24784), 
# we disregard the Kaggle training set and work with the compiled externel DAIGT data set instead.

# In[10]:


daigt_external_data.head()


# # Output next word probability vector using AutoModel

# Sources: [https://huggingface.co/transformers/v3.0.2/model_doc/auto.html](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html)
# [https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text](https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text)
# [https://www.kaggle.com/code/funtowiczmo/hugging-face-transformers-get-started](https://www.kaggle.com/code/funtowiczmo/hugging-face-transformers-get-started)

# In[13]:


import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)


# In[14]:


# Split into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(
    daigt_external_data, test_size=0.2, random_state=42, stratify=daigt_external_data['generated']
)

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")


# In[25]:


#The size of the data above leads to long compute times in processing the data (into vectors of probabilites) and to train the neural network). Let's therefore use a smaller dataset to begin with
small_data = daigt_external_data.sample(n=2400, random_state=42)
train_data, test_data = train_test_split(
    small_data, test_size=400, random_state=42, stratify=small_data['generated']
)

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")


# In[26]:


import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
#Define language model class
class LMHeadModel:

    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    
    def get_next_word_probabilities(self, sentence, top_k=500):

        # Get the model predictions for the sentence.
        predictions = self.get_predictions(sentence)
        
        # Get the next token candidates.
        next_token_candidates_tensor = predictions[0, -1, :]

        # Get the top k next token candidates.
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k).indices.tolist()

        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        
        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))


# In[28]:


# Initialize the language model (e.g., GPT-2)
#list of models to try:     
#model_name = 'gpt2'
#model_name = 'meta-llama/Llama-3.2-1B', 
model_name = 'mistralai/Mistral-7B-v0.1'
#model_name = 'EleutherAI/gpt-neo-125M'
# model_name = 'distilgpt2'
# model_name = 'tiiuae/falcon-7b'
lm_model = LMHeadModel(model_name)


# In[29]:


def process_text(text, lm_model, num_splits=50):
    """
    Process a single text to generate a vector of true next word probabilities.
    """
    # Split the text into words
    words = text.split()
    probabilities = []

    # Ensure there are enough words to sample
    if len(words) < 2:
        return [0.0] * num_splits  # Return zeros if not enough words

    # Generate 50 split points
    for _ in range(num_splits):
        # Randomly select a word index (excluding the last word)
        split_idx = np.random.randint(1, len(words))
        context_words = words[:split_idx]
        true_next_word = words[split_idx]

        # Reconstruct the context sentence
        context_sentence = ' '.join(context_words)

        # Get the next word probabilities
        try:
            next_word_probs = lm_model.get_next_word_probabilities(context_sentence, top_k=500)
        except Exception as e:
            print(f"Error processing context: {e}")
            probabilities.append(0.0)
            continue

        # Find the probability of the true next word
        true_word_prob = 0.0
        for word, prob in next_word_probs:
            if word == true_next_word:
                true_word_prob = prob
                break  # Stop searching once found

        probabilities.append(true_word_prob)

    return probabilities


# In[30]:


# Prepare lists to store the results
X_train = []
y_train = []

# Process training data
print("Processing training data...")
for idx, row in tqdm(train_data.iterrows(), total=len(train_data)):
    text = row['text']
    label = row['generated']
    probs_vector = process_text(text, lm_model, num_splits=10)
    X_train.append(probs_vector)
    y_train.append(label)

# Convert lists to tensors or arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
# Save the numpy arrays to .npy files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

# Similarly process the test data
X_test = []
y_test = []

print("Processing testing data...")
for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
    text = row['text']
    label = row['generated']
    probs_vector = process_text(text, lm_model, num_splits=10)
    X_test.append(probs_vector)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)
# Save the numpy arrays to .npy files
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)


# In[31]:


#Prepare to feed into neural net
#Replace zeros with a small value to avoid issues in log transformation
epsilon = 1e-10
X_train = np.where(X_train == 0, epsilon, X_train)
X_test = np.where(X_test == 0, epsilon, X_test)

# Optionally, apply log transformation
X_train = np.log(X_train)
X_test = np.log(X_test)


# In[32]:


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets and data loaders
batch_size = 32

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# In[34]:


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Binary classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# In[35]:


input_size = X_train.shape[1]
model = FeedForwardNN(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[37]:


#Train the network
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average loss over the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


# In[39]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[40]:


# Randomly select 10 indices from 0 to 49 without replacement
selected_indices = np.random.choice(50, size=10, replace=False)

# Select the probabilities at the chosen indices
X_test_adjusted = X_test[:, selected_indices]

# Proceed with evaluation using X_test_adjusted
X_test_tensor = torch.tensor(X_test_adjusted, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create a new DataLoader for the adjusted test data
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# In[ ]:


#Evaluate model
from sklearn.metrics import accuracy_score, classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))


# Using only gpt-2 and 10 split points per text, we get the following results: Test Accuracy: 0.6225
# Classification Report:
#               precision    recall  f1-score   support
# 
#            0     0.7027    0.5752    0.6326       226
#            1     0.5535    0.6839    0.6118       174
# 
#     accuracy                         0.6225       400
#    macro avg     0.6281    0.6296    0.6222       400
# weighted avg     0.6378    0.6225    0.6236       400

# In[ ]:


get_ipython().system('jupyter nbconvert --to script detect_ai_text_pipeline_CARC.ipynb')


# Sanat's code is below:

# In[23]:


print(kaggle_train['text'][0])


# In[24]:


sentence1, sentence2, sentence3 = kaggle_train['text'][0][:59], kaggle_train['text'][0][:350], kaggle_train['text'][0][:749]


# In[25]:


model = LMHeadModel('bert-base-cased')
model.get_next_word_probabilities(sentence1, top_k=500)


# In[26]:


class LLM_model:
    def __init__(self, name, size, with_prompt = False):
        self.name = name
        self.max_input_len = size
        self.with_prompt = with_prompt


# # Split Training Text into Segments

# In[27]:


#Generates a random integer partition of (a, b)

def rand_part(a, b):
    part = []
    x = randint(a, b)
    part += min(x, b)
    if part[-1]==b:
        return part
    else:
        part.extend(rand_part(x, b))


# In[28]:


import random
def split_txt(begin, essay, max_seq_len = 512):
    segments = []
    y = min(len(essay)-1, begin + max_seq_len - 1)
    x = random.randint(begin, y)
    if essay[x] == ' ':
        #segments.append(essay[:x])
        segments.append(x)
    else:
        while x < y and essay[x] != ' ':
                x+=1
        if x == y and segments == []:
            while essay[x-1] != ' ':
                x-=1
            #segments.append(essay[:x-1])
            segments.append(x-1)
            return segments
        if x == y and segments != []:
            return segments
        else:
            #segments.append(essay[:x])
            segments.append(x)
    segments.extend(split_txt(x+1, essay))
    return segments
        
                


# Example:

# In[29]:


split_txt(0, kaggle_train['text'][0])


# In[30]:


#what's the actual next word in the essay

from string import punctuation

def true_next_word(essay, n):
    word= ''
    i = n
    while str.isalpha(essay[i]) == False and str.isnumeric(essay[i]) == False:
        if essay[i] in punctuation:
            word+=essay[i]
            return word
        else:
            i+=1
    while essay[i] != ' ' and not(essay[i] in punctuation):
        word+=essay[i]
        i+=1
    return word


# Example:

# In[31]:


true_next_word(kaggle_train['text'][0], 59)


# In[32]:


#homemade 'return index of element if it exists' function

def return_index(element, list):
    i=0
    while i < len(list):
        if list[i] == element:
            return i
        else:
            i+=1
    if i == len(list):
        return -10000
    
#get all rth elements of a list of tuples

def rths(r, list):
    rths = []
    for i in range(len(list)):
        if type(list[i]) is tuple:
            if len(list[i]) > r:
                rths.append(list[i][r])
            else:
                rths.append('')
        else:
            rths.append('')
    return rths
            


# # Prediction with prompt and source text using GPT2

# In[33]:


essay_response = """
In recent years, there has been a notable shift in urban planning, with an increasing focus on limiting car usage as a means to foster sustainable and environmentally friendly communities. This shift is evident in various parts of the world, as seen in the case of Vauban, Germany, where an experimental car-free community has thrived since its completion in 2006 (Rosenthal, 2009). Vauban's success challenges the conventional reliance on cars in suburban areas and serves as a model for smart planning that is gaining traction globally.

Vauban's innovative approach to urban development is part of a broader movement to reduce the environmental impact of cars, particularly in suburban settings where car-centric lifestyles have long been the norm. According to experts, passenger cars contribute significantly to greenhouse gas emissions, with Europe attributing 12 percent of emissions to this source, and in some car-intensive areas in the United States, the contribution climbs to a staggering 50 percent (Rosenthal, 2009). Recognizing the environmental implications, planners worldwide are reimagining suburbs, moving away from the traditional car-centric model.

One significant aspect of the shift in urban planning is the concept of "smart planning," where suburbs are designed to be more compact and accessible to public transportation, reducing the need for extensive parking spaces (Rosenthal, 2009). The Vauban model encourages walking and cycling, with essential amenities placed within walking distance along main streets, challenging the conventional suburban sprawl and promoting a more sustainable lifestyle.

The movement toward limiting car usage is not limited to Europe. In Paris, the detrimental impact of car emissions on air quality led to the implementation of a partial driving ban during periods of intense smog (Duffer, 2014). The success of such initiatives is evident in the significant reduction of congestion and improvement in air quality. Similarly, Bogota, Colombia, has embraced a car-free day annually, encouraging alternative transportation methods and reducing both traffic jams and smog levels (Selsky, 2002). The success of these initiatives underscores the potential benefits of limiting car usage in diverse urban settings.

In the United States, there is a growing awareness of the need to reduce car dependency. Recent studies suggest that Americans are buying fewer cars and driving less, indicating a potential shift in cultural attitudes toward car ownership (Rosenthal, 2013). This change aligns with efforts to decrease carbon emissions, as transportation remains a major contributor to the nation's environmental footprint.

While the trend toward limiting car usage presents challenges for the traditional automotive industry, it also opens avenues for innovation and adaptation. Companies like Ford and Mercedes are rebranding themselves as "mobility" companies, recognizing the evolving needs and preferences of consumers (Rosenthal, 2013). The younger generation, in particular, shows a reduced interest in car ownership, preferring alternative modes of transportation facilitated by technological advancements such as car-sharing programs and ride-sharing apps.

In conclusion, the advantages of limiting car usage extend beyond environmental benefits to encompass improved urban planning, reduced congestion, and a shift toward more sustainable lifestyles. The success stories of car-free communities in Germany, driving bans in Paris during smog episodes, and annual car-free days in Bogota demonstrate the feasibility and positive outcomes of such initiatives. As the world grapples with the environmental impact of car culture, embracing alternative transportation models becomes imperative for creating healthier, more livable communities.
"""


# In[34]:


model1 = LLM_model('gpt2', 4096, with_prompt = True)


# In[35]:


def predict_w_prompt(text, prompt, ai):
    split_text = split_txt(0, text, ai.max_input_len)
    probability = 0
    model = LMHeadModel(f'{ai.name}')
    for splice in split_text:
        if ai.with_prompt:
            feed = prompt + text[:splice]
        else:
            feed = text[:splice]
        p_words = model.get_next_word_probabilities(feed[max(0, len(feed)-ai.max_input_len -1):], top_k=500)
        index = return_index(true_next_word(text, splice), rths(0, p_words))
        print(f'index: {index}')
        if index >= 0:
            probability += p_words[index][1]
            print(f'value: {p_words[index][1]}')
        else:
            probability += 1e-10
        print(f'probability: {probability}')
    output = probability/len(split_text)
    return output


# In[36]:


prompt = kaggle_prompts['source_text'][0]+kaggle_prompts['instructions'][0]


# In[37]:


predict_w_prompt(essay_response, prompt, model1)


# In[38]:


model0 = LLM_model('gpt2', 4096, with_prompt = False)


# In[39]:


predict_w_prompt(essay_response, prompt, model0)


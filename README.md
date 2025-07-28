# LLM---Detect-AI-Generated-Text

Farhad de Sousa and Sanat Mulay 

We investigate various approaches to detect AI generated text. Given a prompt and an essay as input, we want to determine whether the essay was written by a human or by an LLM. 

We explore a baseline strategy based on beaming, and conduct a survey of the effect of temperature on next-word generation. We find that our most successful strategy uses an LLM to embed the text and then trains a classifier to classify the embeddings. 

Datasets:
Kaggle dataset (smaller, and contains very few human-written samples)
DAIGT dataset (larger and more diverse)

Methods:

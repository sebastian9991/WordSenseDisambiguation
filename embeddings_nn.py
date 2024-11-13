import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Initialize BERT model and tokenizer
#We use this for dynamic embeddings
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert = BertModel.from_pretrained(bert_model_name)

print("CUDA Information:\n")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
print("End CUDA information.")
'''
Custom dataset to format our data
'''
class WordSenseDataset(Dataset):
    def __init__(self, words, context_sentences, sense_sentences):
        self.words = words
        self.context_sentences = context_sentences
        self.sense_sentences = sense_sentences

    def __len__(self):
        return len(self.words)
    
    '''
    Structure required for dataloader 
    '''
    def __getitem__(self, idx):
        word_embedding = get_bert_embedding(self.words[idx])
        context_embedding = get_bert_embedding(self.context_sentences[idx])
        sense_embedding = get_bert_embedding(self.sense_sentences[idx])
        
        #We concatenate the word and context embeddings for our first layer
        concat_embedding = torch.cat((word_embedding, context_embedding), dim=1)
        #concat is input embedding and sense is our target embedding
        return concat_embedding.squeeze(0), sense_embedding.squeeze(0)
    
'''
We define our own score function on Cosine similarity (That is 1 - sim) for loss
'''
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
    def forward(self, predicted, target):
        cosine_sim = F.cosine_similarity(predicted, target, dim = 1)
        loss = 1 - cosine_sim
        return loss.mean()

'''
BERT is used to give our embeddings dynamically. 
Create a fully-connected NN with one hidden layer. 
'''
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=1536, output_dim=768):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) #Recommended for reguralization 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


'''
Generate BERT embedding for a given text.
https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
'''
def get_bert_embedding(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = bert(**encoded)
    return output.last_hidden_state[:, 0, :]
from torch.utils.data import Dataset, DataLoader

'''
Pre-process data specific for the FNN
'''
def pre_process_FNN(data):
    words = []
    context_sentences = []
    sense_sentences = []
    for element in data:
        words.append(element['word'])
        context_sentences.append(' '.join(element['context']))
        #Sense sentences, we use the examples() if it does not exist default to definition
        #This may be improved at a later date
        if element['label'].examples() != []:
            sense_sentences.append(element['label'].examples()[0])
        else: 
            sense_sentences.append(element['label'].definition())

    
    assert(len(words) == len(context_sentences))
    assert(len(sense_sentences) == len(context_sentences))
    return words, context_sentences, sense_sentences 

'''
Train the model
'''
def train(model, data):

    words, context_sentences, sense_sentences = pre_process_FNN(data)

    # Initialize dataset and dataloader
    dataset = WordSenseDataset(words, context_sentences, sense_sentences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Loss and optimizer
    criterion = CosineSimilarityLoss() #We defined a specific cosine loss here
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_embedding, target_embedding in dataloader:
            optimizer.zero_grad()
            predicted_embedding = model(input_embedding)
            loss = criterion(predicted_embedding, target_embedding)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

'''
Predicts the sense_sentence embeddings; returns score (cosine similarity) (not 1 - loss)
'''
def predict(model, word, context_sentence, sense_sentence):
    model.eval()

    word_embedding = get_bert_embedding(word)
    context_embedding = get_bert_embedding(context_sentence)
    sense_embedding = get_bert_embedding(sense_sentence)

    input_embedding = torch.cat((word_embedding, context_embedding), dim = 1)

    with torch.no_grad():
        predicted_sense_embedding = model(input_embedding)
    
    criterion = CosineSimilarityLoss()

    return  1 - criterion(predicted_sense_embedding, sense_embedding)



    
    

model = FeedForwardNN()

def call_train(data):
    train(model, data)

def call_predict(data):
    words, context_sentences, sense_sentences = pre_process_FNN(data)
    sum = 0
    for word, context_sentence, sense_sentence in zip(words, context_sentences, sense_sentences):
        sum += predict(model, word, context_sentence, sense_sentence)
    return sum / len(data)
    

# In[1]:
import pandas as pd

# In[2]:
df = pd.read_csv("IMDB Dataset.csv")

# In[3]:
df.shape

# In[4]:
df.head()

# In[5]:
df.isnull().sum()

# In[6]:
df.drop_duplicates(inplace=True)

# In[7]:
df.shape

# ## Pre-processing

# ### 1. Converting to lowercase

# In[8]:
df["review"] = df["review"].str.lower()

# ### 2. Removing the URLs

# In[9]:
import re

# sample_text = "abc is the word, abc" # abc => xyz

# new_text = re.sub("abc", "xyz", sample_text)

# In[10]:
def remove_urls(text):
    text = re.sub(r"http\S+" , "", text)  # (pattern, repl, string) eg - https://www.google.com
    return text
    
df["review"] = df["review"].apply(remove_urls)

# ### 3. Removing punctuations

# In[11]:
def remove_punctuations(text):
    text = re.sub(r"[^A-Za-z0-9\s]" , "", text) # A-Z a-z 0-9 \s
    return text

df["review"] = df["review"].apply(remove_punctuations)

# In[12]:
df.head()

# ### 4. Removing HTML

# In[13]:
def remove_html(text):
    text = re.sub(r"<.*?>" , "", text)
    return text

df["review"] = df["review"].apply(remove_html)

# ### 5. Removing the Stopwords

# In[14]:
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# In[15]:
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# In[16]:
# sample_text = "I like coding in python!"
# tokens = word_tokenize(sample_text)

# In[17]:
def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text = text.replace(word, "")

    return text

df["review"] = df["review"].apply(remove_stopwords)

# In[18]:
df.head()

# ### 6. Stemming

# In[19]:
# running -> run
# played -> play
# PorterStemming

from nltk.stem import PorterStemmer

# In[20]:
def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []

    tokens = word_tokenize(text)
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    return " ".join(stemmed_words)

df["review"] = df["review"].apply(stemming)

# In[21]:
df.head()

# ### 7. Encoding

# In[22]:
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["sentiment"] = le.fit_transform(df["sentiment"])

# In[23]:
y = df["sentiment"]

# In[24]:
y

# ### 8. Vectorization

# In[25]:
df.head()

# In[26]:
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(max_features=5000)

X = tf.fit_transform(df["review"])

# ## Dataset & Data Loaders

# In[27]:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In[28]:
X_train.shape

# In[29]:
X_test.shape

# In[30]:
import torch
from torch.utils.data import TensorDataset, DataLoader

# In[31]:
X_train = X_train.toarray()
X_test = X_test.toarray()

# In[32]:
train_set = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float()
)

test_set = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float()
)

# In[33]:
train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=True, batch_size=64)

# ## Build our RNN

# In[34]:
import torch.nn as nn
import torch.optim as optim

# In[35]:
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # optional => shape (num of layers, batch size, hidden size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0) 
        # 1st value = hidden state of all the timesteps => (batch, seq_len, hidden size)
        # 2nd value = final hidden state of last timestep

        out = self.fc(out[:, -1, :])
        return out

# In[36]:
input_size = X_train.shape[1]

model = RNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# ## Training the RNN

# In[37]:
epochs = 10

for epoch in range(epochs):
    model.train()

    for Xb, yb in train_loader:
        optimizer.zero_grad()

        Xb = Xb.unsqueeze(1) # add singleton direction
        
        outputs = model(Xb) # (batch_size, 1)

        outputs = torch.sigmoid(outputs.squeeze()) # (batch_size,) => probability

        loss = criterion(outputs, yb) # compute loss
        loss.backward() # backprop
        optimizer.step() # weights update

    print(f"epoch = {epoch+1}/{epochs} and loss = {loss.item()}")

# In[38]:
# evaluate

model.eval()

with torch.no_grad():
    correct_vals = 0
    tot_vals = 0
    
    for Xb, yb in test_loader:
        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()

        tot_vals += yb.size(0)
        correct_vals += (predicted == yb).sum().item()

    print(f"accuracy = {correct_vals/tot_vals*100}")

# In[39]:

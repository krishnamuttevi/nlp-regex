import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

import spacy

from textblob import TextBlob

import stanza

import flair
import nltk
nltk.download('all')

# Sample text
text = "The quick brown foxes were jumping over the lazy dogs."

# 1. NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

tokens_nltk = word_tokenize(text)
pos_nltk = pos_tag(tokens_nltk)
lemmatizer = WordNetLemmatizer()
lemmatized_nltk = [lemmatizer.lemmatize(word) for word in tokens_nltk]

print("NLTK:")
print("Tokens:", tokens_nltk)
print("POS:", pos_nltk)
print("Lemmatized:", lemmatized_nltk)
print("\n")

# 2. SpaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens_spacy = [token.text for token in doc]
pos_spacy = [(token.text, token.pos_) for token in doc]
lemmatized_spacy = [token.lemma_ for token in doc]

print("SpaCy:")
print("Tokens:", tokens_spacy)
print("POS:", pos_spacy)
print("Lemmatized:", lemmatized_spacy)
print("\n")

# 3. TextBlob
blob = TextBlob(text)
tokens_textblob = blob.words
pos_textblob = blob.tags
lemmatized_textblob = [word.lemmatize() for word in tokens_textblob]

print("TextBlob:")
print("Tokens:", tokens_textblob)
print("POS:", pos_textblob)
print("Lemmatized:", lemmatized_textblob)
print("\n")

# 4. Stanza
stanza.download("en")
nlp_stanza = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma")
doc_stanza = nlp_stanza(text)
tokens_stanza = [word.text for sent in doc_stanza.sentences for word in sent.words]
pos_stanza = [(word.text, word.xpos) for sent in doc_stanza.sentences for word in sent.words]
lemmatized_stanza = [word.lemma for sent in doc_stanza.sentences for word in sent.words]

print("Stanza:")
print("Tokens:", tokens_stanza)
print("POS:", pos_stanza)
print("Lemmatized:", lemmatized_stanza)
print("\n")

# 5. Flair
flair_sentence = flair.data.Sentence(text)
flair_pos_tagger = flair.models.SequenceTagger.load("pos")
flair_pos_tagger.predict(flair_sentence)
tokens_flair = [token.text for token in flair_sentence]
pos_flair = [(token.text, token.get_tag("pos").value) for token in flair_sentence]
lemmatized_flair = "Flair does not provide direct lemmatization."

print("Flair:")
print("Tokens:", tokens_flair)
print("POS:", pos_flair)
print("Lemmatized:", lemmatized_flair)

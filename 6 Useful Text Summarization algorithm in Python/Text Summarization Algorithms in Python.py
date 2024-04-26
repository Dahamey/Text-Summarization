#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;">6 Handy Text Summarization Algorithms in Python</h1>

# <u>**Author</u> :** [Younes Dahami](https://www.linkedin.com/in/dahami/)

# # Introduction
# 
# Excited about Python algorithms that can condense vast amounts of text into concise, insightful summaries? Get set for a thrilling dive into text summarization with Python, where words turn into meaningful insights in no time. In this comprehensive guide, we'll unveil the secrets of one of NLP's most captivating applications.
# 
# This notebook is our portal to mastering the art of extracting crucial information from mountains (heaps) of data. Join us as we delve into algorithms, explore top-notch libraries, and simplify the entire process, step by step. By journey's end, you'll have the power to transform lengthy articles, research papers, and documents into concise, digestible jewels.
# 
# ![alt](text_summarization.png)

# Text summarization involves two distinct scenarios : **Extractive** and **Abstractive** :
# 
# **Extractive Text Summarization :** involves extracting important information from large volumes of text and organizing it into concise summaries. This method selects text based on factors such as the text to be summarized, the most crucial sentences (Top K), and the importance of each sentence to the overall topic. However, this approach is limited to predefined parameters, which can sometimes result in biased extracted text.
# 
# **Abstractive Text Summarization :** involves generating readable sentences from the <u>entire text input</u>. It rewrites large volumes of text by creating appropriate representations, which are then analyzed and summarized using natural language processing. What sets this approach apart is its AI-like ability to understand text using a machine's semantic capabilities and refine it using NLP techniques.
# 

# "Extractive text summarization" is commonly used by automated text summarizers due to its simplicity in many scenarios. Meanwhile, the "abstractive summarization" method, although not as straightforward to use as the extractive one, offers significant benefits in many cases. In many ways, it serves as a precursor to fully automated AI authoring tools. However, this doesn't mean that extractive summarization is obsolete.
# 
# ![alt](extractive_abstractive.png)

# # 6 Techniques for Text Summarization
# 
# Below are 6 approaches to text summarization, incorporating both abstractive and extractive methods.

# ### 1) SUMY
# 
# `Sumy` is a library and command line utility designed for extracting summaries from HTML pages or plain texts. It offers various algorithms for summarization, such as LSA, Luhn, Edmundson, and others.
# 
# * **Latent Semantic Analysis (LSA) :** also known as Latent Semantic Indexing (LSI), is a technique in natural language processing and information retrieval used for analyzing relationships between a set of documents and the terms they contain. LSA aims to capture the underlying structure of the text corpus by representing words and documents in a high-dimensional semantic space. In text summarization, LSA can be used to identify important concepts and relationships within the text, allowing for the generation of concise summaries that capture the main ideas.
# 
# 
# * **Luhn Algorithm :** proposed by Hans Peter Luhn in 1958, is a statistical approach to automatic text summarization. This algorithm assigns a score to each sentence in a document based on the frequency of important words or phrases. The basic idea behind the Luhn Algorithm is to prioritize sentences that contain a high frequency of significant terms, assuming that these sentences are more likely to convey essential information. In text summarization, the Luhn Algorithm is used to identify and select sentences with the most relevant content for inclusion in the summary.
# 
# 
# * **Edmundson Algorithm :** introduced by Harold Edmundson in 1969, is another statistical method for automatic text summarization. Similar to the Luhn Algorithm, the Edmundson Algorithm assigns scores to sentences based on the occurrence of specific keywords or phrases. However, the Edmundson Algorithm also incorporates additional features, such as word location and syntactic cues, to determine the importance of sentences. By considering various factors, including keyword frequency, position, and grammatical structure, the Edmundson Algorithm aims to produce more informative and coherent summaries. In text summarization, the Edmundson Algorithm is utilized to identify important sentences that effectively capture the main points of the document.
# 

# These algorithms represent different approaches to text summarization, each with its strengths and limitations. Depending on the characteristics of the text and the requirements of the summarization task, one algorithm may be more suitable than others for generating concise and informative summaries.

# Here's an example of how to use Sumy with the LSA algorithm :

# In[1]:


#!pip install sumy


# In[2]:


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import warnings
warnings.filterwarnings('ignore')

# Input text to be summarized
input_text = """
Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.

One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.

Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.

Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.

Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.

Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.

In summary, natural language processing and text summarization are essential areas of research with broad applications in fields such as information retrieval, document summarization, question answering, sentiment analysis, and conversational agents. By enabling computers to understand and process human language, NLP holds the promise of revolutionizing how we interact with technology and access information in the digital age.
"""

# Parse the input text
parser = PlaintextParser.from_string(input_text, Tokenizer("english"))

# Create an LSA summarizer
summarizer = LsaSummarizer()

# Generate the summary : # we can adjust the number of sentences in the summary
summary = summarizer(parser.document, sentences_count = 3)

# Output the summary
print(f"Original Text :\n{input_text}")
print("\nSummary :\n")
for sentence in summary :
    print(sentence)


# ### 2) BERT Extractive Summarization
# 
# BERT, which stands for **Bidirectional Encoder Representations from Transformers**, is a state-of-the-art natural language processing (NLP) model developed by researchers at Google AI Language. It represents a significant advancement in NLP by leveraging the Transformer architecture and pre-training techniques to generate contextualized word embeddings.
# 
# Key features of BERT include:
# 
# * **Bidirectional Context :** BERT is designed to capture bidirectional context by jointly training on both left-to-right (forward) and right-to-left (backward) directions. This allows BERT to understand the context of each word based on the entire input sequence, enabling more accurate representations of word meanings.
# 
# 
# * **Transformer Architecture :** BERT is built upon the Transformer architecture, which consists of self-attention mechanisms and feed-forward neural networks. This architecture enables BERT to efficiently model long-range dependencies in text and capture intricate linguistic patterns.
# 
# 
# * **Pre-training** and **Fine-tuning :** BERT is pre-trained on large corpora of text data using unsupervised learning objectives, such as masked language modeling and next sentence prediction. After pre-training, BERT can be fine-tuned on downstream NLP tasks with labeled data to adapt its representations for specific tasks, such as sentiment analysis, named entity recognition, and question answering.
# 
# 
# * **Contextualized Embeddings :** BERT produces contextualized word embeddings, meaning that the representation of each word depends on its surrounding context within the input sequence. This allows BERT to generate more accurate and contextually rich representations of words, leading to better performance on various NLP tasks.
# 

# BERT has achieved state-of-the-art results on a wide range of NLP benchmarks and tasks, demonstrating its effectiveness in understanding and processing natural language text. Its availability in various pre-trained versions, such as BERT-base and BERT-large, has made it widely adopted in both research and industry for various NLP applications.
# 
# ![alt](BERT.png)

# **BERT** can also be utilized for extractive summarization. In this approach, sentences are ranked according to their significance, and the top-ranked sentences constitute the summary. The `bert-extractive-summarizer` library offers a straightforward interface for BERT-based extractive summarization.

# In[3]:


#!pip install bert-extractive-summarizer


# Here’s an example of how to use BERT for extractive summarization :

# In[4]:


from summarizer import Summarizer

# Input text to be summarized
input_text = """
Le domaine de l'apprentissage automatique (AA) est une branche passionnante de l'intelligence artificielle (IA) qui consiste à développer des algorithmes capables d'apprendre à partir de données pour effectuer des tâches spécifiques sans être explicitement programmés. L'une des applications les plus fascinantes de l'AA est le résumé automatique de texte, qui vise à réduire la taille d'un texte tout en conservant ses informations essentielles.

Le résumé automatique de texte peut être réalisé de différentes manières, notamment par extraction et par abstraction. Dans la méthode d'extraction, les phrases ou paragraphes les plus pertinents sont identifiés et extraits du texte source pour former un résumé. En revanche, dans la méthode d'abstraction, un modèle d'AA génère de nouveaux mots, phrases ou paragraphes qui capturent l'essence du texte source.

Les modèles de résumé automatique de texte reposent souvent sur des techniques avancées de traitement du langage naturel (TLN) et de Deep Learning. Les réseaux de neurones récurrents (RNR) et les réseaux neuronaux convolutionnels (RNC) sont couramment utilisés pour traiter les données textuelles et générer des résumés de qualité. De plus, les modèles basés sur les transformers, tels que BERT, GPT et T5, ont révolutionné le résumé automatique de texte en utilisant des mécanismes d'attention pour capturer les relations contextuelles entre les mots et les phrases.

Cependant, le résumé automatique de texte pose plusieurs défis, notamment la préservation de la cohérence, de la précision et du sens du texte original. Les modèles doivent être capables de comprendre le contexte et les nuances du langage humain pour produire des résumés qui sont à la fois concis et informatifs.

En plus du résumé automatique de texte extractive et abstractive, il existe également d'autres approches, telles que la résumé par clustering, qui regroupe les phrases similaires pour former des résumés, et la résumé par classification, qui classe les phrases en fonction de leur importance pour créer un résumé.

L'impact de l'AA sur le résumé automatique de texte est significatif dans de nombreux domaines, notamment la recherche académique, le journalisme, la veille stratégique, et même l'aide à la prise de décision dans les entreprises. Les systèmes de résumé automatique permettent de traiter de grandes quantités d'informations rapidement et efficacement, ce qui peut être particulièrement utile dans un monde où l'information est omniprésente et en constante expansion.

Cependant, malgré les avancées remarquables, le résumé automatique de texte n'est pas encore parfait et reste un domaine de recherche actif. Des défis persistants, tels que la génération de résumés fidèles au contenu original, la gestion des biais et des opinions, ainsi que l'adaptation à différents types de textes et de langues, nécessitent des efforts continus de la part des chercheurs et des praticiens.
"""

# Create a BERT extractive summarizer
summarizer = Summarizer()

# Generate the summary : we can adjust the 'min_length' and 'max_length' parameters
summary = summarizer(input_text,
                    min_length = 60,
                    max_length = 170)

# Output the summary
print(f"Original Text :\n{input_text}")
print(f"\nSummary :\n {summary}")


# ### 3) BART Abstractive Summarization
# 
# "In addition to extractive summarization, **BART** can also be employed for abstractive summarization. Here's how you can utilize BART for abstractive summarization using the `transformers` library :

# In[5]:


# "pt": Return PyTorch tensors.
# "tf": Return TensorFlow tensors.
# "np": Return NumPy arrays.


# In[6]:


from transformers import BartTokenizer, BartForConditionalGeneration

# load pre-trained BART model & tokenizer
pretrained_model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path = pretrained_model_name)
model = BartForConditionalGeneration.from_pretrained(pretrained_model_name)

# Input text to be summarized
input_text = """
Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.

One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.

Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.

Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.

Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.

Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.

In summary, natural language processing and text summarization are essential areas of research with broad applications in fields such as information retrieval, document summarization, question answering, sentiment analysis, and conversational agents. By enabling computers to understand and process human language, NLP holds the promise of revolutionizing how we interact with technology and access information in the digital age.
"""

# Tokenize & summarize the input text using BART
inputs = tokenizer.encode("Summarize: " + input_text,
                          return_tensors = "pt", # pytorch tensors
                          max_length = 1025,
                          truncation = True)
print("Length of Tokenized Input:", len(inputs[0]))
summary_ids = model.generate(inputs,
                              max_length = 150,
                              min_length = 60,
                              length_penalty = 2.0,
                              num_beams = 4,
                              early_stopping = True)


# Decode & output the summary 
summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
print(f"Original Text :\n{input_text}")
print(f"\nSummary :\n {summary}")


# * **`length_penalty` :** This parameter is used to modify the length of the generated sequences during decoding. It encourages the model to generate sequences with a certain length. A higher value for length_penalty encourages longer sequences, while a lower value encourages shorter ones. Typically, length_penalty is set to a value greater than 1 to encourage longer sequences. In our example, length_penalty = 2.0 means that the model slightly favors longer sequences.
# 
# 
# * **`num_beams` :** Beam search is a technique used in sequence generation tasks where the model considers multiple candidate sequences during decoding. `num_beams` specifies the number of beams (or candidate sequences) that the model should consider. A higher value for `num_beams` increases the diversity of the generated sequences but also increases computation time. In our example, num_beams = 4 means that the model considers four candidate sequences during decoding.
# 
# These parameters are important for controlling the trade-off between the quality and diversity of the generated summaries. Adjusting them can help in obtaining summaries that better match the desired characteristics.

# ### 4) T5 Abstractive Summarization
# 
# **T5 (Text-to-Text Transfer Transformer)** is a versatile transformer model suitable for various NLP tasks, including summarization. Here's how we can utilize T5 for abstractive summarization using the `transformers` library :
# 

# In[7]:


#!pip install SentencePiece


# In[11]:


from transformers import T5Tokenizer, T5ForConditionalGeneration

# load pre-trained T5 model & tokenizer
pretrained_model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path = pretrained_model_name)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)

# Input text to be summarized
input_text = """
Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.

One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.

Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.

Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.

Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.

Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.

In summary, natural language processing and text summarization are essential areas of research with broad applications in fields such as information retrieval, document summarization, question answering, sentiment analysis, and conversational agents. By enabling computers to understand and process human language, NLP holds the promise of revolutionizing how we interact with technology and access information in the digital age.
"""

# Tokenize & summarize the input text using T5
inputs = tokenizer.encode("Summarize : "+input_text,
                         return_tensors = "pt",
                         max_length = 1025,
                         truncation = True)
print("Length of Tokenized Input:", len(inputs[0]))
summary_ids = model.generate(inputs,
                            max_length = 150,
                            min_length = 60,
                            length_penalty = 2.0,
                            num_beams = 4,
                            early_stopping = True)

# Decode & output the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
print(f"Original Text :\n{input_text}")
print(f"\nSummary :\n {summary}")


# ### 5) Gensim
# 
# **Gensim** is a Python library primarily used for topic modeling and document similarity analysis. It also offers a straightforward implementation of **TextRank**, an unsupervised algorithm based on graph theory.

# **TextRank** is an unsupervised graph-based ranking algorithm that is commonly used for keyword extraction and text summarization tasks. It was introduced as an extension of Google's PageRank algorithm, which ranks web pages based on their importance on the web.
# 
# In TextRank, the text is represented as a graph, where nodes represent individual units of text (such as words or sentences), and edges represent the relationships between them. The algorithm works by iteratively updating the ranking scores of nodes based on the structure of the graph.
# 
# Here's how TextRank works:
# 
# * **Graph Construction :** First, the text is preprocessed to tokenize it into individual units (usually words or sentences). Then, a graph is constructed where each node represents a unit of text, and edges between nodes represent relationships such as co-occurrence or similarity. Typically, the similarity between nodes is computed using metrics like "cosine similarity" or "Jaccard similarity".
# 
# * **Node Weight Calculation :** Each node in the graph is assigned an initial weight based on some criteria, such as the frequency of occurrence of the corresponding word or the importance of the corresponding sentence. These weights represent the importance of each unit of text in the graph.
# 
# * **Iterative Ranking :** The TextRank algorithm iteratively updates the ranking scores of nodes based on the importance of their neighboring nodes. During each iteration, the ranking score of a node is computed as a weighted sum of the ranking scores of its neighbors. The weights are determined by the strength of the relationships between nodes, which can be measured using metrics like "edge weights" or "similarity scores".
# 
# * **Convergence :** The iterative process continues until the ranking scores of the nodes converge to stable values. Typically, convergence is achieved when the difference between successive iterations falls below a predefined threshold.
# 
# * **Ranking :** Once the algorithm converges, the nodes are ranked based on their final ranking scores. The top-ranked nodes are considered to be the most important units of text in the original document and can be used for tasks such as "keyword extraction" or "text summarization".
# 
# TextRank has been widely used in various natural language processing tasks, including keyword extraction, text summarization, and document clustering, due to its simplicity, effectiveness, and unsupervised nature.

# In[35]:


#!pip install "gensim==3.8.2"


# The `ratio` parameter in the summarize function from `gensim.summarization` module controls the length of the generated summary relative to the original text.

# In[23]:


#from gensim.summarization import summarize

# Input text to be summarized
input_text = """
Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.

One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.

Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.

Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.

Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.

Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.

In summary, natural language processing and text summarization are essential areas of research with broad applications in fields such as information retrieval, document summarization, question answering, sentiment analysis, and conversational agents. By enabling computers to understand and process human language, NLP holds the promise of revolutionizing how we interact with technology and access information in the digital age.
"""

# Generate the summary using TextRank algorithm : we can adjust the ratio parameter based on the summary length you desire
#summary = summarize(input_text, ratio = 0.3)

# Output summary
#print(f"Original Text :\n{input_text}")
#print(f"\nSummary :\n {summary}")


# ### 6) TextTeaser
# 
# TextTeaser is an automatic summarization algorithm designed to take an article and generate a summary. It relies on the TextRank algorithm and is effective at producing concise summaries.

# In[33]:


#!pip install textteaser


# In[34]:


#from textteaser import TextTeaser

# Input text to summarize
input_text = """
Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.

One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.

Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.

Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.

Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.

Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.

In summary, natural language processing and text summarization are essential areas of research with broad applications in fields such as information retrieval, document summarization, question answering, sentiment analysis, and conversational agents. By enabling computers to understand and process human language, NLP holds the promise of revolutionizing how we interact with technology and access information in the digital age.
"""

# Create a TextTeaser object
#tt = TextTeaser()

# Generate the summary
#summary = tt.summarize(input_text)

# Output the summary
#print(f"Original Text :\n{input_text}")
#print(f"\nSummary :\n {summary}")


# In[ ]:





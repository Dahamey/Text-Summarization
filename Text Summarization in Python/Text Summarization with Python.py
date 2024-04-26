#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;">Text Summarization with Python</h1>

# <u>**Author</u> :** [Younes Dahami](https://www.linkedin.com/in/dahami/)

# # Introduction
# 
# **Text summarization** involves condensing a document to highlight its key information. The goal is to provide a concise summary of the main points. In this article, I'll present a machine learning project focused on text summarization using Python.

# ![alt](text_summarization.png)
# 
# With the vast amount of digital data available, it's crucial to devise a method for quickly summarizing lengthy texts while preserving the core idea. Text summarization not only helps reduce reading time but also accelerates information retrieval and facilitates obtaining comprehensive insights on a topic.

# The primary objective of employing machine learning for text summarization is to condense the source text into a shorter version while retaining its essential information and meaning. Text summarization entails providing concise descriptions of multiple texts, such as reports generated from one or more documents, conveying relevant knowledge in a fraction of the original text's length.
# 
# Now that you understand what text summarization is and why machine learning is useful for it, let's delve into a machine learning project on text summarization with Python.

#  # Text Summarization with Python
# 
# Next, I'll guide you through the process of text summarization with Python. We'll begin by importing the required Python libraries :

# In[2]:


import nltk
import string
from heapq import nlargest


# We don't need extensive machine learning for this task. We can summarize text without training a model. However, we do need to employ natural language processing, for which I'll utilize the `NLTK` library in Python.

# ## NLTK library:
# 
# 
# The **NLTK (Natural Language Toolkit)** library is a comprehensive platform for building Python programs that work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.
# 
# Key features of NLTK include:
# 
# * **Corpora :** NLTK provides access to numerous text corpora, including those for languages other than English, which are valuable resources for training and testing natural language processing (NLP) models.
# 
# 
# * **Tokenization :** NLTK offers functions for tokenizing text, breaking it down into individual words or sentences, which is a fundamental step in many NLP tasks.
# 
# * **Stopword Removal :** NLTK provides built-in lists of stopwords (commonly occurring words like "the," "is," "and," etc.) in various languages. Removing stopwords from the text can help in focusing on content-bearing words and phrases during summarization.
# 
# * **Stemming :** NLTK includes algorithms for stemming words, reducing them to their base or root forms. This is useful for tasks like text normalization and information retrieval.
# 
# 
# * **Part-of-Speech (POS) Tagging :** NLTK provides tools for assigning parts of speech (e.g., noun, verb, adjective) to words in a sentence, which is crucial for syntactic analysis and semantic understanding.
# 
# 
# * **Parsing :** NLTK supports various parsing algorithms for analyzing the structure of sentences and extracting syntactic relationships between words.
# 
# 
# * **Named Entity Recognition (NER) :** NLTK includes modules for identifying named entities such as people, organizations, and locations in text.
# 
# 
# * **Text Classification :** NLTK facilitates text classification tasks by providing algorithms and tools for feature extraction, model training, and evaluation.
# 
# 
# * **Lexical Resources :** NLTK offers access to lexical resources such as WordNet, which is a large lexical database of English words and their semantic relationships.
# 
# 
# * **Language Processing Modules :** NLTK includes modules for processing text in different languages, supporting tasks such as language detection, translation, and transliteration.
# 
# * **Summarization Algorithms :** NLTK offers implementations of various text summarization algorithms, including extractive and abstractive summarization techniques :
# 
#      - **Extractive summarization methods :** involve selecting important sentences or passages from the original text.
#      - **Abstractive summarization methods :** generate summaries by rewriting the content in a more concise form.

# NLTK is widely used by researchers, educators, and practitioners in the fields of computational linguistics, NLP, and artificial intelligence for prototyping and deploying applications that involve the analysis and understanding of human language. In text summarization, NLTK provides various tools and functionalities that can aid in the process of condensing large volumes of text into shorter, coherent summaries.

# ## String library
# 
# The **string** library in Python provides various constants and functions for working with strings. 

# ## HEAPQ library
# 
# The **heapq** library in Python provides functions for implementing heaps (priority queues) efficiently.
# 
# * **A priority queue :** is a data structure that allows for efficient retrieval of the smallest (or largest) element in a collection of elements.
# 
# Here's an overview of the `heapq` library and the `nlargest` function in the context of text summarization:
# 
# * **`heapq` Library :**
# 
# The heapq library provides several functions for heap manipulation, including :
#    * heapify(iterable): Converts an iterable into a heap in linear time.
#    * heappush(heap, item): Pushes an item onto a heap while maintaining the heap property.
#    * heappop(heap): Removes and returns the smallest item from the heap.
#    * heapreplace(heap, item): Pops and returns the smallest item from the heap, and then pushes the new item.
#    * heappushpop(heap, item): Pushes the item onto the heap, and then pops and returns the smallest item.
# 
# These functions allow us to efficiently maintain a priority queue and retrieve the smallest elements.

# * **`nlargest` Function in Text Summarization :**
# 
# In text summarization, the `nlargest` function from the `heapq` library is often used to extract the most important sentences or phrases from a document.
# This function returns the "$n$" largest elements from an iterable (we used a dictionary in the code below, we called it `sentences_score_dic`), based on a specified key function or criteria.
# In the context of text summarization, the iterable may represent sentences or phrases from the document, and the key function could be based on metrics such as sentence length, TF-IDF scores, or other relevance scores.
# 
# By using `nlargest`, we can efficiently identify and extract the most relevant content for summarization, helping to condense the information while preserving its key points.

# Overall, the `heapq` library and the `nlargest` function are valuable tools for implementing efficient text summarization algorithms by prioritizing and extracting the most important information from a document.

# Now, let's proceed with some steps to remove punctuations from the text, followed by text processing steps. Finally, we'll tokenize the text, and we'll be able to observe the results for text summarization with Python :

# Using the code below, I will summarize this text : 
# 
# **"** *Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.*
# 
# *One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.*
# 
# *Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.*
# 
# *Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.*
# 
# *Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.*
# 
# *Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.*
# 
# *In summary, natural language processing and text summarization are essential areas of research with broad applications in fields such as information retrieval, document summarization, question answering, sentiment analysis, and conversational agents. By enabling computers to understand and process human language, NLP holds the promise of revolutionizing how we interact with technology and access information in the digital age.* **"**

# In[4]:


# Assigning the input text to a variable 'text'
input_text = """
Natural Language Processing (NLP) is a fascinating field that sits at the intersection of linguistics, computer science, artificial intelligence, and information engineering. Its primary aim is to enable computers to understand, interpret, and generate human language in a meaningful way. NLP encompasses a wide range of tasks, including speech recognition, language understanding, sentiment analysis, machine translation, and text summarization.

One of the fundamental challenges in NLP is dealing with the inherent ambiguity and complexity of human language. Unlike programming languages, which have strict syntax and semantics, natural languages like English, Spanish, and Chinese are highly nuanced and context-dependent. This complexity arises from factors such as homonyms, synonyms, idioms, sarcasm, and cultural references, making it challenging for computers to accurately process and analyze text.

Text summarization is a crucial task in NLP, aiming to distill the essential information from a document while preserving its meaning and key insights. There are two main approaches to text summarization: extractive and abstractive. Extractive summarization involves selecting a subset of sentences or phrases from the original text to form the summary, whereas abstractive summarization involves generating new sentences that capture the main ideas of the document in a concise and coherent manner.

Extractive summarization techniques often rely on statistical methods, graph algorithms, or machine learning models to identify the most informative sentences based on criteria such as sentence importance, relevance, and diversity. These techniques may involve ranking sentences using features such as word frequency, sentence length, and semantic similarity, or constructing sentence graphs and applying graph-based algorithms like PageRank or TextRank.

Abstractive summarization, on the other hand, is more challenging and requires advanced natural language generation techniques, such as neural networks and sequence-to-sequence models. These models learn to generate summaries by mapping input sequences (source text) to output sequences (summaries) using deep learning architectures like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models.

Despite significant advancements in NLP and text summarization, many challenges remain. These include handling ambiguity and context, generating coherent and grammatically correct summaries, preserving factual accuracy and avoiding bias, and scaling summarization models to process large volumes of text efficiently. Addressing these challenges requires interdisciplinary research and collaboration across fields such as linguistics, cognitive science, computer science, and artificial intelligence.

"""
# checking if the number of sentences in the text is greater than 20
if input_text.count(". ") > 20 :
    # if the condition is 'true', calculate the length of the summary based on the number of sentences
    length = int(round(text.count(". ")/10, 0))
else :
    # if the condition is 'false', set the length of the summary to 1
    length = 1
    
# Removing punctuation from the text
no_punc_char = [char for char in input_text if char not in string.punctuation]
no_punc_text = ''.join(no_punc_char)

# Tokenizing the processed text and removing stopwords
pro_text = [word for word in no_punc_text.split() if word.lower() not in nltk.corpus.stopwords.words("english")]

# Counting the frequency of each word in the processed text
word_freq_dic = {}
for word in pro_text :
    if word not in word_freq_dic :
        word_freq_dic[word] = 1     # adding the new word to the dictionary 
    else : # if the word already exist in the dictionary
        word_freq_dic[word] +=1

# Normalizing the word frequencies 
max_freq  = max(word_freq_dic.values())
for word in word_freq_dic :
    word_freq_dic[word] /= max_freq
    
# Tokenizing the input text into sentences
sentences_list = nltk.sent_tokenize(input_text)
sentences_score_dic = {}

# Calculating the score for each sentence based on word frequencies
for sentence in sentences_list :
    # word-tokenizing
    for word in nltk.word_tokenize(sentence.lower()) :
        if word in word_freq_dic.keys() :
            if sentence not in sentences_score_dic.keys() :
                sentences_score_dic[sentence] = word_freq_dic[word]
            else :
                sentences_score_dic[sentence] += word_freq_dic[word]
                
                
# Selecting the top-ranked sentences based on their scores
summary_sentences = nlargest(n = length,
                            iterable = sentences_score_dic,
                            key = sentences_score_dic.get)

# Joining the selected sentences to form the summary
summary = " ".join(summary_sentences)

# Printing the sumamry
print(summary)


# ## Le Résumé Automatique de Texte en Français (Text Summarization in French)

# In[5]:


# Assigning the input text to a variable 'text'
input_text = """
Le domaine de l'apprentissage automatique (AA) est une branche passionnante de l'intelligence artificielle (IA) qui consiste à développer des algorithmes capables d'apprendre à partir de données pour effectuer des tâches spécifiques sans être explicitement programmés. L'une des applications les plus fascinantes de l'AA est le résumé automatique de texte, qui vise à réduire la taille d'un texte tout en conservant ses informations essentielles.

Le résumé automatique de texte peut être réalisé de différentes manières, notamment par extraction et par abstraction. Dans la méthode d'extraction, les phrases ou paragraphes les plus pertinents sont identifiés et extraits du texte source pour former un résumé. En revanche, dans la méthode d'abstraction, un modèle d'AA génère de nouveaux mots, phrases ou paragraphes qui capturent l'essence du texte source.

Les modèles de résumé automatique de texte reposent souvent sur des techniques avancées de traitement du langage naturel (TLN) et de Deep Learning. Les réseaux de neurones récurrents (RNR) et les réseaux neuronaux convolutionnels (RNC) sont couramment utilisés pour traiter les données textuelles et générer des résumés de qualité. De plus, les modèles basés sur les transformers, tels que BERT, GPT et T5, ont révolutionné le résumé automatique de texte en utilisant des mécanismes d'attention pour capturer les relations contextuelles entre les mots et les phrases.

Cependant, le résumé automatique de texte pose plusieurs défis, notamment la préservation de la cohérence, de la précision et du sens du texte original. Les modèles doivent être capables de comprendre le contexte et les nuances du langage humain pour produire des résumés qui sont à la fois concis et informatifs.

En plus du résumé automatique de texte extractive et abstractive, il existe également d'autres approches, telles que la résumé par clustering, qui regroupe les phrases similaires pour former des résumés, et la résumé par classification, qui classe les phrases en fonction de leur importance pour créer un résumé.

L'impact de l'AA sur le résumé automatique de texte est significatif dans de nombreux domaines, notamment la recherche académique, le journalisme, la veille stratégique, et même l'aide à la prise de décision dans les entreprises. Les systèmes de résumé automatique permettent de traiter de grandes quantités d'informations rapidement et efficacement, ce qui peut être particulièrement utile dans un monde où l'information est omniprésente et en constante expansion.

Cependant, malgré les avancées remarquables, le résumé automatique de texte n'est pas encore parfait et reste un domaine de recherche actif. Des défis persistants, tels que la génération de résumés fidèles au contenu original, la gestion des biais et des opinions, ainsi que l'adaptation à différents types de textes et de langues, nécessitent des efforts continus de la part des chercheurs et des praticiens.

"""

# checking if the number of sentences in the text is greater than 20
if input_text.count(". ") > 20 :
    # if the condition is 'true', calculate the length of the summary based on the number of sentences
    length = int(round(input_text.count(". ")/10, 0))
else :
    # if the condition is 'false', set the length of the summary to 1
    length = 1
    
# Removing punctuation from the text
no_punc_char = [char for char in input_text if char not in string.punctuation]
no_punc_text = ''.join(no_punc_char)

# Tokenizing the processed text and removing stopwords
pro_text = [word for word in no_punc_text.split() if word.lower() not in nltk.corpus.stopwords.words("french")]

# Counting the frequency of each word in the processed text
word_freq_dic = {}
for word in pro_text :
    if word not in word_freq_dic :
        word_freq_dic[word] = 1     # adding the new word to the dictionary 
    else : # if the word already exist in the dictionary
        word_freq_dic[word] +=1

# Normalizing the word frequencies 
max_freq  = max(word_freq_dic.values())
for word in word_freq_dic :
    word_freq_dic[word] /= max_freq
    
# Tokenizing the input text into sentences
sentences_list = nltk.sent_tokenize(input_text, language = "french")
sentences_score_dic = {}

# Calculating the score for each sentence based on word frequencies
for sentence in sentences_list :
    # word-tokenizing
    for word in nltk.word_tokenize(sentence.lower()) :
        if word in word_freq_dic.keys() :
            if sentence not in sentences_score_dic.keys() :
                sentences_score_dic[sentence] = word_freq_dic[word]
            else :
                sentences_score_dic[sentence] += word_freq_dic[word]
                
                
# Selecting the top-ranked sentences based on their scores
summary_sentences = nlargest(n = length,
                            iterable = sentences_score_dic,
                            key = sentences_score_dic.get)

# Joining the selected sentences to form the summary
summary = " ".join(summary_sentences)

# Printing the sumamry
print(summary)


# ## الملخص التلقائي للنصوص بالعربية (Text Summarization in Arabic)

# In[6]:


# Assigning the input text to a variable 'text'
input_text = """
مجال التعلم الآلي (AA) هو فرع مثير في مجال الذكاء الاصطناعي (IA) يهدف إلى تطوير الخوارزميات قادرة على التعلم من البيانات لأداء مهام محددة دون برمجتها صراحةً. أحد أهم تطبيقات التعلم الآلي هو ملخص النصوص التلقائي، الذي يهدف إلى تقليل حجم النص مع الحفاظ على معلوماته الأساسية.

يمكن تحقيق ملخص النصوص تلقائيًا بطرق مختلفة، بما في ذلك الاستخراج والتجريد. في الطريقة الاستخراجية، يتم تحديد واستخراج الجمل أو الفقرات الأكثر صلة من النص المصدر لتكوين ملخص. بينما في الطريقة الابتكارية، يقوم نموذج AA بإنشاء كلمات، جمل أو فقرات جديدة تلخص جوهر النص المصدر.

يعتمد نماذج ملخص النصوص التلقائي غالبًا على تقنيات متقدمة في معالجة اللغة الطبيعية (TLN) والتعلم العميق. وتُستخدم الشبكات العصبية العائدة (RNR) والشبكات العصبية التحويلية (RNC) بشكل شائع لمعالجة البيانات النصية وإنشاء ملخصات عالية الجودة. بالإضافة إلى ذلك، ثورة في مجال ملخص النصوص التلقائي باستخدام آليات الانتباه لالتقاط العلاقات السياقية بين الكلمات والجمل.

ومع ذلك، يواجه ملخص النصوص التلقائي عدة تحديات، بما في ذلك الحفاظ على التماسك والدقة والمعنى الأصلي للنص. يجب أن تكون النماذج قادرة على فهم السياق والتفاصيل الدقيقة للغة البشرية لإنتاج ملخصات موجزة ومعلوماتية.

بالإضافة إلى الملخص التلقائي الاستخراجي والابتكاري، هناك أيضًا نهج آخر مثل الملخص بالتجميع، الذي يجمع الجمل المماثلة لتكوين ملخص، والملخص بالتصنيف، الذي يصنف الجمل بناءً على أهميتها لإنشاء ملخص.

يتمثل تأثير التعلم الآلي على ملخص النصوص التلقائي في مجالات عديدة مثل البحث الأكاديمي، والصحافة، والمراقبة الاستراتيجية، وحتى المساعدة في اتخاذ القرارات في الشركات. تسمح أنظمة الملخص التلقائي بمعالجة كميات كبيرة من المعلومات بسرعة وكفاءة، مما يمكن أن يكون مفيدًا بشكل خاص في عالم يتسم بوجود المعلومات في كل مكان وباستمرار.

ومع ذلك، على الرغم من التقدم الملحوظ، فإن ملخص النصوص التلقائي ليس مثاليًا بعد ويظل مجالًا نشطًا للبحث. التحديات المستمرة، مثل إنشاء ملخصات موافقة للمحتوى الأصلي، وإدارة التحيزات والآراء، وكذلك التكيف مع مختلف أنواع النصوص واللغات، تتطلب جهودًا مستمرة من الباحثين والممارسين.

"""

# checking if the number of sentences in the text is greater than 20
if input_text.count(".") > 10 :
    # if the condition is 'true', calculate the length of the summary based on the number of sentences
    length = int(round(input_text.count(". ")/5, 0))
else :
    # if the condition is 'false', set the length of the summary to 1
    length = 1
    
# Removing punctuation from the text
no_punc_char = [char for char in input_text if char not in string.punctuation]
no_punc_text = ''.join(no_punc_char)

# Tokenizing the processed text and removing stopwords
pro_text = [word for word in no_punc_text.split() if word.lower() not in nltk.corpus.stopwords.words("arabic")]

# Counting the frequency of each word in the processed text
word_freq_dic = {}
for word in pro_text :
    if word not in word_freq_dic :
        word_freq_dic[word] = 1     # adding the new word to the dictionary 
    else : # if the word already exist in the dictionary
        word_freq_dic[word] +=1

# Normalizing the word frequencies 
max_freq  = max(word_freq_dic.values())
for word in word_freq_dic :
    word_freq_dic[word] /= max_freq
    
# Tokenizing the input text into sentences
sentences_list = nltk.sent_tokenize(input_text)
sentences_score_dic = {}

# Calculating the score for each sentence based on word frequencies
for sentence in sentences_list :
    # word-tokenizing
    for word in nltk.word_tokenize(sentence.lower()) :
        if word in word_freq_dic.keys() :
            if sentence not in sentences_score_dic.keys() :
                sentences_score_dic[sentence] = word_freq_dic[word]
            else :
                sentences_score_dic[sentence] += word_freq_dic[word]
                
                
# Selecting the top-ranked sentences based on their scores
summary_sentences = nlargest(n = length,
                            iterable = sentences_score_dic,
                            key = sentences_score_dic.get)

# Joining the selected sentences to form the summary
summary = " ".join(summary_sentences)

# Printing the sumamry
print(summary)


# # Change Log
# 
# | Date (DD-MM-YYYY) | Version | Changed By      | Change Description      |
# | ----------------- | ------- | -------------   | ----------------------- |
# | 26-04-2024       | 1.0     | Younes Dahami   |  initial version |
# 

# In[ ]:





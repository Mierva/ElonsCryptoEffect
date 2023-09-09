from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from gensim import corpora, utils, models, matutils
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
import spacy


class TextVectorizer:
    """
    Vectorization of raw texts into numerical form.
    """    
    def __init__(self, verbose=False) -> None:
        self.logger = logging.getLogger("complex_method")

        if self.logger.hasHandlers():
            # Remove existing handlers to prevent accumulation
            self.logger.handlers.clear()

        if verbose: 
            # Create a console handler for displaying log messages
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)  # Set the logging level to DEBUG for more detailed output
    
    def __lemmatization(self, texts: list[str], allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):    
        if type(texts)!=list:
            texts = [texts]
        
        self.logger.info("Lem1.")
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        url_pattern = [{"label": "URL",
                        "pattern": [{"LIKE_URL": True}]}]

        ruler = nlp.add_pipe('entity_ruler', before='ner')
        ruler.add_patterns(url_pattern)

        self.logger.info("Lem2.")
        texts_out = []
        for text in texts:
            # TODO: consider using nlp.pipe which should be faster
            doc = nlp(text)
            
            cleaned_text = []
            for token in doc:
                if token.ent_type_ != 'URL' and not token.is_stop and token.pos_ in allowed_postags:
                    cleaned_text.append(token.lemma_)                    
                    
            final = ' '.join(cleaned_text)
            texts_out.append(final)

        return texts_out

    def __create_ngrams(self, texts: list[str]):
        data_words = []
        self.logger.info("Ngram1.")
        for text in texts:
            tokenized_text = utils.simple_preprocess(text)
            data_words.append(tokenized_text)
            
        self.logger.info("Ngram2.")
        bigrams_phrases  = models.Phrases(data_words, min_count=3, threshold=50)
        trigrams_phrases = models.Phrases(bigrams_phrases[data_words], threshold=50)

        bigram  = models.phrases.Phraser(bigrams_phrases)
        trigram = models.phrases.Phraser(trigrams_phrases)
        
        self.logger.info("Ngram3.")
        data_bigrams = [bigram[doc] for doc in data_words]
        data_bigrams_trigrams = [trigram[bigram[doc]] for doc in data_bigrams]
        
        return data_bigrams_trigrams

    def vectorize_texts(self, texts_ngrams:list[list[str]]):
        """
        ## Args:
            texts_ngrams (list[list[str]]): made by create_ngrams method.

        ## Returns:
            tuple: contains id2word and corpus
        """        
        self.logger.info("Vec1.")
        id2word = corpora.Dictionary(texts_ngrams)
        corpus = [id2word.doc2bow(text) for text in texts_ngrams]
        
        return id2word, corpus
    
    def prepare_tfidf(self, raw_texts: pd.Series, lemmatized_texts):       
        lemmatized_texts = self.__lemmatization(raw_texts)
        data_bigrams_trigrams = self.__create_ngrams(lemmatized_texts)
        id2word, corpus = self.vectorize_texts(data_bigrams_trigrams)

        tfidf = models.TfidfModel(corpus, id2word=id2word)
        tfidf_vectorizer = TfidfVectorizer(max_df=0.6,
                                           min_df=5,
                                           ngram_range=(1,3))
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(lemmatized_texts)

        id2word = dict((v, k) for k, v in tfidf_vectorizer.vocabulary_.items())
        corpus = matutils.Sparse2Corpus(tfidf_matrix.T)

        low_value = 0.03
        words = []
        words_missing_in_tfidf = []
        for i in range(0, len(corpus)):
            bow = corpus[i]
            low_value_words = []
            tfidf_ids = [id for id,_ in tfidf[bow]]
            bow_idf = [id for id,_ in bow]
            low_value_words = [id for id, value in tfidf[bow] if value < low_value]
            
            drops = low_value_words + words_missing_in_tfidf
            for item in drops:
                words.append(id2word[item])
            
            # words with tfidf score of 0 will be missing
            words_missing_in_tfidf = [id for id in bow_idf if id not in tfidf_ids]
            new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
            corpus[i] = new_bow
            
        return id2word, corpus

    def make_pipeline(self, steps=None):
        """
        ## Args:
            steps (list[tuples], optional): list of preprocessing steps to be applied.\n
            if not specified then all steps are included. 

        ## Returns:
            sklearn.pipeline.Pipeline: Pipeline containing methods as transformers.
        """        
        if steps==None:
            steps = [('lemmatization', self.__lemmatization),
                    ('trigrams', self.__create_ngrams),
                    ('vectorization', self.vectorize_texts)]

        for i, step in enumerate(steps):
            steps.insert(i, (step[0], FunctionTransformer(step[1])))
            steps.remove(step)

        return Pipeline(steps)
    
    def prepare_new_texts(self, texts: list[str], id2word):
        """For preprocessing new text for prediction.\n

        ## Args:
            texts (list[str]): raw texts from tweets.\n
            id2word (Dicitonary): dictionary based on all seen texts.

        ## Returns:
            list: texts represented in numeric form.
        """        
        steps = [('lemmatization', self.__lemmatization),
                ('trigrams', self.__create_ngrams)]

        corpus_pipeline = self.make_pipeline(steps)
        texts_ngrams = corpus_pipeline.transform(texts)
        corpus = [id2word.doc2bow(text) for text in texts_ngrams]
    
        return corpus
    
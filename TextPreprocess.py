from sklearn.preprocessing import FunctionTransformer
from gensim import corpora, utils, models
from sklearn.pipeline import Pipeline
import spacy


class TextPreprocessor:
    """
    Preprocessor of raw texts into numerical form.
    """
    def __init__(self, texts: list[str]):
        self.texts = texts
    
    def lemmatization(self, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):    
        if type(self.texts)!=list:
            self.texts = [self.texts]
        
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        url_pattern = [{"label": "URL",
                        "pattern": [{"LIKE_URL": True}]}]

        ruler = nlp.add_pipe('entity_ruler', before='ner')
        ruler.add_patterns(url_pattern)

        
        texts_out = []
        for text in self.texts:
            # TODO: consider using nlp.pipe which should be faster
            doc = nlp(text)
            
            cleaned_text = []
            for token in doc:
                if token.ent_type_ != 'URL' and not token.is_stop and token.pos_ in allowed_postags:
                    cleaned_text.append(token.lemma_)
                    
            final = ' '.join(cleaned_text)
            texts_out.append(final)

        return texts_out

    def create_ngrams(self):
        data_words = []
        for text in self.texts:
            tokenized_text = utils.simple_preprocess(text)
            data_words.append(tokenized_text)

        bigrams_phrases  = models.Phrases(data_words, min_count=3, threshold=50)
        trigrams_phrases = models.Phrases(bigrams_phrases[data_words], threshold=50)

        bigram  = models.phrases.Phraser(bigrams_phrases)
        trigram = models.phrases.Phraser(trigrams_phrases)

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
        id2word = corpora.Dictionary(texts_ngrams)
        corpus = [id2word.doc2bow(text) for text in texts_ngrams]
        
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
            steps = [('lemmatization', self.lemmatization),
                    ('trigrams', self.create_ngrams),
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
        steps = [('lemmatization', self.lemmatization),
                ('trigrams', self.create_ngrams)]

        corpus_pipeline = self.make_pipeline(steps)
        texts_ngrams = corpus_pipeline.transform(texts)
        corpus = [id2word.doc2bow(text) for text in texts_ngrams]
    
        return corpus
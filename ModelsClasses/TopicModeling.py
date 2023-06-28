from gensim.models import LdaMulticore, CoherenceModel
from gensim.test.utils import datapath
from bayes_opt import BayesianOptimization


class TopicModeling(LdaMulticore):
    """Implementation of gensim LDA model with typical fit-predict methods.
    """    
    
    # best_model = read_parameters_from_file
    best_model = 0
    
    def black_box_function(self, x, y):
        """Function with unknown internals we wish to maximize.

        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        return -x ** 2 - (y - 1) ** 2 + 1
    
    
    def tune_model(self, params_grid: dict, texts, verbose=2):
        # TODO: implement bayesian tuning
        models_scores = {}

        # models = [self.fit(params_grid, i) for i in range(95,170,5)]
        
        # for lda_model in models:
        #     coherence_model_lda = CoherenceModel(model=lda_model, 
        #                                         texts=texts, 
        #                                         corpus=params_grid['corpus'], 
        #                                         dictionary=params_grid['id2word'])
            
        #     coherence_score = coherence_model_lda.get_coherence() 
        #     models_scores.update({lda_model: coherence_score})
        
        
        pbounds = {'x': (2, 4), 'y': (-3, 3)}
        optimizer = BayesianOptimization(
            f=LdaMulticore,
            pbounds=pbounds,
            verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
        optimizer.maximize(init_points=2,
                           n_iter=3)
        
        for i, res in enumerate(optimizer.res):
            print(f"Iteration {i}: \n\t{res}")
            
        optimizer.set_bounds(new_bounds={"x": (-2, 3)})
        optimizer.maximize(init_points=0,
                           n_iter=5)
        
        optimizer.probe(params={"x": 0.5, "y": 0.7},
                        lazy=True)
                    
        return models_scores

    # def fit(self, params_grid, num_topics):
    #     lda_model = LdaMulticore(corpus=params_grid['corpus'],
    #                              num_topics=num_topics,
    #                              id2word=params_grid['id2word'],
    #                              random_state=1,
    #                              passes=10,
    #                              per_word_topics=True)
        
    #     return lda_model
    
    # def predict(self, data):
    #     predicts = data.apply(lambda x: dict(self.get_document_topics(x, minimum_probability=0)))
        
    #     return predicts
        
    def update(self, new_texts):
        # Update model with unseen data.
        other_texts = [['computer', 'time', 'graph'],
                       ['survey', 'response', 'eps'],
                       ['human', 'system', 'computer']]
        
        dictionary = 'read_dictionary_from_file'
        new_corpus = [dictionary.doc2bow(text) for text in new_texts]
        self.update(new_corpus)
        
        
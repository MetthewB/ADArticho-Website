from flair.data import Sentence
from flair.models import SequenceTagger

import pandas as pd

class PosTagger():

    def __init__(
            self, 
            caption_df : pd.DataFrame, 
            caption_index : str = 'caption', 
            image_index : str = 'image_id', 
            counter_file_path : str = 'data/pos_counter.csv',
            no_inference=False
        ):

        if not no_inference:
            self.__tagger = SequenceTagger.load("flair/pos-english")

        self.__caption_df = caption_df
        self.__caption_index = caption_index
        self.__image_index = image_index
        self.__no_inference = no_inference
        
        self.__counter_file_path = counter_file_path

        self.__label_dict = {
            'ADD' : 9, 'AFX' : 9, 'CC' : 8, 'CD' : 9, 'DT' : 3, 'EX' : 6, 'FW' : 9, 'HYPH' : 9, 'IN' : 2, 'JJ' : 1, 'JJR' : 1, 'JJS' : 1, 'LS' : 9, 
            'MD' : 5, 'NFP' : 9, 'NN' : 0, 'NNP' : 0, 'NNS' : 0, 'NNPS' : 0, 'PDT' : 9, 'POS' : 9,  'PRP' : 4, 'PRP$' : 4, 'RB' : 6, 'RBR' : 6, 
            'RBS' : 6, 'RP' : 9, 'SIM' : 9, 'TO' : 9, 'UH' : 7, 'VB' :	5, 'VBD' : 5, 'VBG' : 5, 'VBN' : 5, 'VBP' : 5, 'VBZ' : 5, 'WDT' : 9, 
            'WP' : 4, 	 'WP$' : 4, 	 'WRB' : 6, 'XX' : 9, '.' : 9, ',' : 9,
        }


    def __add_sentence_pos(self, sentence : str, counter : list[int]) -> list[int]:
        if len(counter) != 10 :
            raise ValueError()
    
        sentence = Sentence(sentence)

        self.__tagger.predict(sentence)

        for s in sentence :
            counter[self.__label_dict.setdefault(s.to_dict()['labels'][0]['value'], 9)] += 1

        return counter
    
    
    def __count_sentence_pos_for_image(self, image_id : int):
        counter = [0,0,0,0,0,0,0,0,0,0]
        captions = self.__caption_df.loc[self.__caption_df[self.__image_index] == image_id, self.__caption_index]

        for caption in captions:
            counter = self.__add_sentence_pos(caption, counter)

        return counter
    
    
    def __load_counters_dataframe(self,):

        try :
            counter_df = pd.read_csv(self.__counter_file_path)
        except:

            counter_df = pd.DataFrame(self.__caption_df[self.__image_index].drop_duplicates())
            counter_df['noun'] = 0
            counter_df['adjective'] = 0
            counter_df['preposition'] = 0
            counter_df['article'] = 0
            counter_df['pronoun'] = 0
            counter_df['verb'] = 0
            counter_df['adverb'] = 0
            counter_df['interjection'] = 0
            counter_df['conjunction'] = 0
            counter_df['unknown'] = 0

            counter_df.to_csv(self.__counter_file_path, index=False)
    
        return counter_df
    
    def count_sentence_pos_for_images(self, images : list[int]):
        if self.__no_inference:
            raise RuntimeError('If no_inference is set to true, no inference can be performed !')

        counter_df = self.__load_counters_dataframe()

        i = 1
        for image_id in images:
            print("Processing image : " + str(image_id))
            counter = self.__count_sentence_pos_for_image(image_id)

            counter_df.loc[counter_df['image_id'] == image_id, ['noun','adjective','preposition','article','pronoun','verb','adverb','interjection','conjunction','unknown']] = counter
    
            if i % 10 == 0 :
                counter_df.to_csv(self.__counter_file_path, index=False)

            i+=1
            
        counter_df.to_csv(self.__counter_file_path, index=False)

    def compute_fscore(self):
        counter_df = self.__load_counters_dataframe()

        counter_df['nf'] = counter_df['noun'] + counter_df['adjective'] + counter_df['preposition'] + counter_df['article']
        counter_df['nc'] = counter_df['pronoun'] + counter_df['verb'] + counter_df['adverb'] + counter_df['interjection'] #+ counter_df['unknown']
        counter_df['f-score'] =  50*(1 + (counter_df['nf'] - counter_df['nc'])/(counter_df['nf'] + counter_df['nc'] + counter_df['conjunction']))

        return pd.DataFrame(counter_df[['image_id', 'f-score']])

    def counters(self,):
        return self.__load_counters_dataframe().copy()

    def text_f_score(self, sentence):
        counter = self.__add_sentence_pos(sentence, [0,0,0,0,0,0,0,0,0,0])
        nf = counter[0] + counter[1] + counter[2] + counter[3]
        nc = counter[4] + counter[5] + counter[6] + counter[7]
        return 50*(1 + (nf-nc)/(nf+nf+counter[8]))
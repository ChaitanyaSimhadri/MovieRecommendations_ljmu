import sys
from dataclasses import dataclass


import numpy as np 
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer




from projectApp.exception import CustomException
from projectApp.logger import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os

from projectApp.utils import save_object
from projectApp.utils import get_data_transformer_object1
from projectApp.utils import get_data_transformer_object2,get_data_transformer_object3,get_data_transformer_object4


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('projectApp','artifacts',"proprocessor.pkl")
    #preprocessor_obj_file_path2=os.path.join('artifacts',"proprocessor2.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    
        
    def initiate_data_transformation(self, df):
        try:
            new_df = pd.read_csv(df)
            logging.info("Reading data completed")
            logging.info("Obtaining preprocessing object")

             

            
            logging.info("Applying preprocessing object on  dataframe.")
            
            new_df['keywords'] = new_df['keywords'].apply(get_data_transformer_object1)
            new_df['genres'] = new_df['genres'].apply(get_data_transformer_object1)
            logging.info("Applied preprocessing object on  dataframe keywords,genres done.")

            new_df['production_companies'] = new_df['production_companies'].apply(get_data_transformer_object2)
            new_df['cast'] = new_df['cast'].apply(get_data_transformer_object2)
            new_df['crew'] = new_df['crew'].apply(get_data_transformer_object4)
            new_df['spoken_languages'] = new_df['spoken_languages'].apply(get_data_transformer_object2)
            new_df['original_language'] = new_df['original_language'].apply(get_data_transformer_object4)
            logging.info("Applied preprocessing object 2 on remaining dataframe.")
            # ... Apply other transformations as needed

            new_df['overview'] = new_df['overview'].apply(lambda x: x.split())
            new_df['overview'] = new_df['overview'].apply(get_data_transformer_object3)
            logging.info("Applied preprocessing object 3 overview.")

            tags = new_df['overview']+ new_df['keywords'] + 10* new_df['genres'] + new_df['production_companies'] + new_df['cast']  + new_df['crew'] + new_df['spoken_languages']
            
            tags = tags.apply(lambda x: " ".join(x))

            tfidf = TfidfVectorizer(max_features=5000,stop_words='english')

            tags_vector =  tfidf.fit_transform(tags)
            
            logging.info("Saved preprocessing object.")
            
            # Save preprocessing objects as pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=(get_data_transformer_object1,get_data_transformer_object2,tfidf)
            )

          
            return tags_vector, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
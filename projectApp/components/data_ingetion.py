import os
import sys
from projectApp.exception import CustomException
from projectApp.logger import logging
import pandas as pd,numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import ast
from projectApp.utils import weighted_rating,convert

from projectApp.components.data_transformation import DataTransformation
from projectApp.components.data_transformation import DataTransformationConfig

from projectApp.components.model_trainer import ModelTrainerConfig
from projectApp.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    
    raw_data_path: str=os.path.join('projectApp','artifacts',"data.csv")
    tags_path = os.path.join('projectApp','artifacts',"tags.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        self.convert1 =  lambda obj: [i['name'] for i in ast.literal_eval(obj)]
        self.get_cast = lambda obj: [i['name'] for i in ast.literal_eval(obj) if i['order'] <= 3]
        self.get_directors = lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director']

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df1=pd.read_csv('projectApp\data\\tmdb_5000_credits.csv')
            df2=pd.read_csv('projectApp\data\\tmdb_5000_movies.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df= df1.merge(df2,on = 'title', how = 'left')

            df['genres'] = df['genres'].apply(convert)
            df['keywords'] = df['keywords'].apply(convert)
            df['production_companies'] = df['production_companies'].apply(convert)
            df['spoken_languages'] = df['spoken_languages'].apply(convert)
            df['cast'] = df['cast'].apply(self.get_cast)
            df['crew']=df['crew'].apply(self.get_directors)
            df['year'] = pd.to_datetime(df['release_date']).dt.year
            df['month'] = pd.to_datetime(df['release_date']).dt.month
            df['weighted_rating'] = weighted_rating(df[['vote_count','vote_average']],np.mean(df['vote_average']),df['vote_count'].quantile(0.75))
           
            features = ['movie_id','title','release_date','keywords','overview','original_language','genres', 'production_companies', 'cast','crew', 'spoken_languages','popularity','vote_count','budget','revenue', 'runtime','vote_average','weighted_rating']
            tag_features = ['movie_id','title','keywords','overview','original_language','genres','production_companies', 'cast','crew', 'spoken_languages']
            

            df = df[features]
            
            tags_df=df[tag_features]
            tags_df = tags_df.dropna()
            

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            tags_df.to_csv(self.ingestion_config.tags_path,index=False,header=True)

        
            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.tags_path
                

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data,tags_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    tags,_ = data_transformation.initiate_data_transformation(tags_data)

    modeltrainer=ModelTrainer()
    similiarity,_ = modeltrainer.initiate_model_trainer(tags)

    
    
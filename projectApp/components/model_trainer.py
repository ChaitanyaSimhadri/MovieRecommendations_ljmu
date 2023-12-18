import os
import sys
from dataclasses import dataclass



from projectApp.exception import CustomException
from projectApp.logger import logging

from projectApp.utils import save_object

from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('projectApp',"artifacts","similarity.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,tags_vector):
        try:
            logging.info("Reading tags vecotor")
            
            similarity = cosine_similarity(tags_vector)

            ## To get best model name from dict

            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=similarity
            )

           
            return similarity,self.model_trainer_config.trained_model_file_path

            
        except Exception as e:
            raise CustomException(e,sys)
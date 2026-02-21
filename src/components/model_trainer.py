import os
import sys
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__ (self):
        self.model_trainer= ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        logging.info(f"Model trainer initiated")
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info(f"Train and Test dataset spilt into X and y")

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            logging.info(f"Hyperparameter tuning initiated")
            
            params = {
                'Linear Regression' : {},
                'Lasso' : {
                    'alpha': [0.01, 0.1, 1.0, 5.0, 10.0]
                    },
                    
                'Ridge' : {
                    'alpha': [0.01, 0.1, 1.0, 5.0, 10.0]
                    },
                    
                'K-Neighbors Regressor' : {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                    },
    
                'Decision Tree' : {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                    },

                'Random Forest Regressor' : {
                    'n_estimators': [100, 200, 300],
                    'max_features': [None, 'sqrt', 'log2'],
                    'max_depth': [None, 10, 20]
                    },
    
                'XGBRegressor' : {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                    },
                'CatBoosting Regressor' : {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 500]
                    },
    
                'AdaBoost Regressor' : {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                    }
                    }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(model_score)]
            best_model= models[best_model_name]

            if model_score<0.65:
                raise CustomException("No best model found.")
            
            save_object(file_path=self.model_trainer.model_path,
                        obj=best_model)
            logging.info(f"Best model found AND model training is completed")
            
            y_pred = best_model.predict(X_test)
            score = r2_score(y_pred, y_test)
            
            return score

        except Exception as e:
            raise CustomException(e,sys)
"""
This is a boilerplate pipeline 'defenses'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import run_all_defenses
from typing import List

def defense_generation_templete() -> Pipeline:
    return pipeline(
        [
            node(
                func=run_all_defenses,
                inputs=["adversarial_data","model" ,"params:data_params"],
                outputs=["Report","Distributions","KL_Divergence","Confusion_Matrix"],
                name="run_all_defenses",
            )
        ],
            inputs=["adversarial_data","model"],
            outputs=["Report","Distributions","KL_Divergence","Confusion_Matrix"],
            tags=["defense_generation"]
    )
def create_pipeline(attack_types:List[str]=["DeepFool", "CarliniL2","FSGM","PGD"],
                    models:List[str]=["Resnet_model","Regnet_x_model","Regnet_y_model"]) -> Pipeline:

    for index, model_ref in enumerate(models):
        defense_pipeline = [
            pipeline(pipe=defense_generation_templete(),
                     parameters= {"params:data_params":"params:Report_params"},
                     inputs={"model":f"{model_ref}",
                            "adversarial_data":f"{model_ref}_Adversarial_{attack_type}@Dataset"},
                    outputs={"Report":f"{model_ref}_Report_{attack_type}@Dataset",
                             "Distributions":f"{model_ref}_Distributions_{attack_type}@Plot",
                             "KL_Divergence":f"{model_ref}_KL_Divergence_{attack_type}@Dataset",
                             "Confusion_Matrix":f"{model_ref}_Confusion_Matrix_{attack_type}@Plot"},
                     namespace=f"{model_ref}_Defense_Generation_{attack_type}",tags=["defense_generation",attack_type])
                     for attack_type in attack_types
        ]
        if index ==0:
            final_pipeline= sum(defense_pipeline)
        else:
            final_pipeline += sum(defense_pipeline)
    return final_pipeline 

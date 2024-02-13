"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import Train_model,plot_results

def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline([
        node(
            func=Train_model,
            inputs=["train_loader","test_loader","params:model_options"],
            outputs=["model",
                     "report"],
            name = "Training_of_model",
            tags=["training"]
            ),
        
            node(
                func=plot_results,
                inputs="report",
                outputs="figure",
                name="plot_training_results",
                tags=["training"]
                ),
        ],outputs=["model","figure"])

    Resnet_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_Resnet"},
        inputs={"train_loader":"Dataset_for_train_normalized","test_loader":"Dataset_for_test_normalized"},
        outputs={"model":"Resnet_model",
                "figure": "Resnet_plot_results"},
        namespace="Resnet_pipeline"
    ) 
        
    RegnetX_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetX"},
        inputs={"train_loader":"Dataset_for_train_normalized","test_loader":"Dataset_for_test_normalized"},
        outputs={"model":"Regnet_x_model",
                "figure": "Regnet_x_plot_results"},
        namespace="Regnet_x_pipeline"
    )

    RegnetY_pipeline = pipeline(
        pipe = pipeline_instance,
        parameters={"params:model_options":"params:training_parameters_RegnetY"},
        inputs={"train_loader":"Dataset_for_train_normalized","test_loader":"Dataset_for_test_normalized"},
        outputs={"model":"Regnet_y_model",
                "figure": "Regnet_y_plot_results"},
        namespace="Regnet_y_pipeline"
    )

    return Resnet_pipeline+RegnetX_pipeline+RegnetY_pipeline



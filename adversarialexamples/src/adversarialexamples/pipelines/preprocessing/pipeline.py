"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import create_loader

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node(
        #     func=compute_mean_std,
        #     inputs="CIFAR10_train",
        #     outputs=["mean_data", "std_data"],
        #     name="compute_mean_std_node",
        # ),
        node(
            func=create_loader,
            inputs=["Dataset_for_train", "params:Parameters_training"],
            outputs="Dataset_for_train_normalized",
            name="create_loader_train_node",
            tags=['training']
        ),
        node(
            func=create_loader,
            inputs=["Dataset_for_test", "params:Parameters_testing"],
            outputs="Dataset_for_test_normalized",
            name="create_loader_test_node",
            tags=['training', 'adversarial_generation']
        ),
        
    ])

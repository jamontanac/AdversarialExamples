from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.project import settings
import logging
from pathlib import Path
from kedro.pipeline import Pipeline

class MyHooks:
    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        self._logger.info(f"Catalog created with elements :{catalog.list()}")
    #In case the model has been trained, skip the training process
    # @hook_impl
    # def before_pipeline_run(self, run_params, pipeline, catalog):
    #     # Specify the path to your model file
    #     # print(pipeline)
    #     if "training" in run_params['tags'] and len(run_params['tags']) == 1:
            
    #         node_list = pipeline.nodes
    #         final_nodes = []
    #         # Check if the model file exists
    #         models_to_check = ["Resnet_model", "Regnet_x_model", "Regnet_y_model"]

            # for model in models_to_check:
    #             if catalog.exists(model):
    #                 print(f"delete model {model}")
    #                 models_to_check.remove(model)
    #             if catalog.exists(models_to_check[index]):
    #                 print(f"delete model {models_to_check[index]} and namespace {namespaces[index]}")
    #                 models_to_check.remove(models_to_check[index])
    #                 namespaces.remove(namespaces[index])
            
    #         # preprocessing_pipeline = pipeline.filter()
    #         # print(preprocessing_pipeline)
    #         for node in node_list:
    #             if node._func_name == 'create_loader':
    #                 final_nodes.append(node)
    #             if node._func_name == 'Train_model' and node.outputs[0] in models_to_check:
    #                 final_nodes.append(node)
    #             if node._func_name== 'plot_results' and node.namespace in namespaces:
    #                 final_nodes.append(node)
    #         print(final_nodes)
    #         modified_pipeline = Pipeline(final_nodes)
    #         print(modified_pipeline)
    #         # print(modified_pipeline)
    #         # If the model exists, create a modified pipeline that excludes the training step
    #         return modified_pipeline  # This modified pipeline will be used for the run

    #     # If the model doesn't exist, the original pipeline will run as is
    #     return pipeline

    # @hook_impl
    # def before_node_run(self, catalog: DataCatalog, node) -> None:
    #     if node._func_name == 'train_model':
    #         if catalog.exists('Regnet_x_model'):
    #             self._logger.info(f"Model already exists, skipping training")
    #             node.skip()

    # @hook_impl
    # def after_context_created(self, context:KedroContext) -> None:
    #     current_env = context.env
    #     conf_source = f'{settings.CONF_SOURCE}/{current_env}'
    #     config_loader = OmegaConfigLoader(conf_source)
    #     print(config_loader.get('catalog'))

                          
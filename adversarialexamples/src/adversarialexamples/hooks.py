from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.project import settings
import logging

class MyHooks:
    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        self._logger.info(f"Catalog created with elements :{catalog.list()}")
                          
    # @hook_impl
    # def after_context_created(self, context:KedroContext) -> None:
    #     current_env = context.env
    #     conf_source = f'{settings.CONF_SOURCE}/{current_env}'
    #     config_loader = OmegaConfigLoader(conf_source)
    #     print(config_loader.get('catalog'))

                          
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from kedro.io import AbstractDataset
import plotly.graph_objects as go
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from kedro_datasets.matplotlib import MatplotlibWriter
from kedro_datasets.plotly import JSONDataset

class ImageFolderDataSet(AbstractDataset):
    def __init__(self, filepath: str):
        """
        Creates a new instance of ImageFolderDataSet to save images into folders.
        :param filepath: The path to the directory where the folders and images will be saved.
        """
        self._filepath = Path(filepath)

    def _save(self, data: Dict[str, Dict[str, Figure]]) -> None:
        """
        Saves images from the provided dictionary into folders.
        :param data: A dictionary where keys are folder names and values are dictionaries
                     of image names and matplotlib figures.
        """
        for folder , images in data.items():
            folder_path = self._filepath / folder
            folder_path.mkdir(parents=True, exist_ok=True)

            for img_name, figure in images.items():
                img_path = folder_path / f"{img_name}.jpg"
                figure.savefig(img_path, format='jpg')
                # If the figure is not automatically closed:
                plt.close(figure)

    def _load(self) -> None:
        """
        The load operation is not supported in this dataset.
        """
        raise NotImplementedError("Load operation is not supported.")

    def _describe(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the dataset.
        """
        return dict(filepath=str(self._filepath))
class CustomMatplotlibWriter(MatplotlibWriter):
    def __init__(self, filepath: str, **kwargs):
        """
        Creates a new instance of CustomMatplotlibWriter to save images into folders.
        :param filepath: The path to the directory where the folders and images will be saved.
        """
        super().__init__(filepath=filepath, **kwargs)
        self._filepath = Path(filepath)
    def _save(self, data: Dict[str, Dict[str, plt.Figure]]) -> None:
        """
        Saves images from the provided dictionary into folders.
        :param data: A dictionary where keys are folder names and values are dictionaries
                     of image names and matplotlib figures.
        """
        for folder , images in data.items():
            folder_path = self._filepath / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            for image_name, fig in images.items():
                # Construct the full path for each image
                image_path = folder_path / f"{image_name}"
                # Save the figure using matplotlib's savefig
                fig.savefig(image_path)
                # Close the figure to free memory
                plt.close(fig)

class CustomJSONDataSetPlotly(JSONDataset):
    def __init__(self, filepath: str,**kwargs):
        """
        Creates a new instance of PlotlyImageFolderDataSet to save Plotly figures into folders.
        :param filepath: The path to the directory where the folders and images will be saved.
        """
        super().__init__(filepath=filepath,**kwargs)
        self._base_filepath = Path(filepath)

    def _save(self, data: Dict[str, go.Figure]) -> None:
        """
        Saves Plotly figures from the provided dictionary into folders.
        :param data: A dictionary where keys are folder names and values are dictionaries
                     of image names and Plotly figures.
        """
        for folder, figure in data.items():
            folder_path = self._base_filepath / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            img_path = folder_path / f"{folder}.png"
            figure.write_image(str(img_path))
            # super()._save(data=figure,filepath=str(img_path))
            

    def _load(self) -> None:
        """
        The load operation is not supported in this dataset.
        """
        raise NotImplementedError("Load operation is not supported.")

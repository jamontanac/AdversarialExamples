"""
This is a boilerplate pipeline 'defenses'
generated using Kedro 0.19.1
"""
import matplotlib.pyplot as plt
import matplotlib.figure as matplt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import itertools
import torch.backends.cudnn as cudnn

import torch.optim as optim
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from .tranformation_defense import JPEGTransform, FlipTransform, ResizePadTransform, DistortTransform, ResizePadFlipTransform
import logging
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

class AdversarialDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        """
        Args:
            data_dict (dict): Dictionary containing adversarial data.
            transform (callable, optional): Optional transform to be applied on the examples.
        """
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict["real_labels"])

    def __getitem__(self, idx):
        sample = {key: value[idx] for key, value in self.data_dict.items()}
        if self.transform:
            sample["examples"] = self.transform(sample["examples"])
        return sample




def init_model(model:torch.nn.Module,lr:float=0.001)->Tuple[nn.Module,nn.Module,optim.Optimizer]:
    # Move model to GPU if available
    model = model.to(device)
    if device == 'cuda':
        model= torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model


def General_defense_transform(adversarial_data: Dict, transformations:List, params_data: Dict,batch_size:int = 512) -> torch.utils.data.DataLoader:
    """
    This function applies a general transformation to the adversarial examples
    """
    transform = transforms.Compose([
        *transformations,
        transforms.Normalize(mean=params_data["mean"], std=params_data["std"])
    ])
    dataset = AdversarialDataset(adversarial_data, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def KL_divergence(p,q):
    epsilon = 1e-10
    p += epsilon
    q +=epsilon

    p /= p.sum()
    q /= q.sum()

    return (p*(p/q).log()).sum().item()

def plot_confidence_distribution_single_class(data, class_name):
    with pd.option_context('mode.use_inf_as_na', True):
        g = sns.PairGrid(data, diag_sharey=False)
        g.map_upper(sns.kdeplot, levels=12, cmap="icefire")
        g.map_lower(sns.kdeplot, levels=7, cmap="icefire")
        g.map_lower(sns.scatterplot, s=10)
        g.map_diag(sns.histplot, bins=32, stat="density", element="step")
        g.figure.suptitle(f'{class_name} Distributions')
        plt.tight_layout()
    return g.figure

def plot_confidence_distribution_all(Confidences, Labels, classes):
    figs = {}  # List to store individual figures
    Distances = {}
    confidences_model = Confidences["Original"]
    confidences_adversarial = Confidences["Adversarial"]
    confidences_defense = Confidences["Defense"]
    true_labels = Labels["True_labels"]

    for idx, class_name in enumerate(classes):
        values_model = [confidences_model[i] for i, label in enumerate(true_labels) if label == idx]
        values_adversarial = [confidences_adversarial[i] for i, label in enumerate(true_labels) if label == idx]
        values_defense = [confidences_defense[i] for i, label in enumerate(true_labels) if label == idx]

        model_hist = torch.histc(torch.Tensor(values_model),bins=32,min=0,max=1)
        adv_hist = torch.histc(torch.Tensor(values_adversarial),bins=32,min=0,max=1)
        def_hist = torch.histc(torch.Tensor(values_defense),bins=32,min=0,max=1)
        Distances[f"{class_name}_Adversarial"] = KL_divergence(adv_hist,model_hist)
        Distances[f"{class_name}_Defense"] = KL_divergence(def_hist,model_hist)
        with pd.option_context('mode.use_inf_as_na', True):
            data = pd.DataFrame({"Model": values_model, "Adversarial": values_adversarial, "Defense": values_defense})
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            fig = plot_confidence_distribution_single_class(data, class_name)
        figs[f"{class_name}.png"] = fig
        # figs.append(fig)
    return figs, Distances



def plot_sub_confusion_matrix(fig, cm, classes, position, show_colorbar=False):
    """
    This function adds a heatmap of the confusion matrix to a subplot position.
    """
    
    hm = go.Heatmap(
        z=cm[::-1],
        x=classes,
        y=classes[::-1], 
        colorscale='blues',
        showscale=show_colorbar,
        coloraxis="coloraxis",
        hoverinfo="x+y+z",
        hovertemplate="Predicted: %{x}<br>Real: %{y}<br>Number: %{z}<extra></extra>"  # <extra></extra> hides additional hover info
    )

    fig.add_trace(hm, row=1, col=position)

    for i, row in enumerate(cm):
        for j, val in enumerate(row):
             fig.add_annotation(
                dict(
                    x=classes[j], 
                    y=classes[i], 
                    xref=f'x{position}',
                    yref=f'y{position}',
                    text=f"<b>{val}</b>",
                    showarrow=False,
                    font=dict(
                        color="gray",
                        size=12
                    )
                )
            )
    fig.update_yaxes(title_text="True label", row=1, col=position)
    fig.update_xaxes(title_text="Predicted label", row=1, col=position)

def plot_confusion_matrix(labels,classes):
    # Compute confusion matrices
    conf_matrix_defense = confusion_matrix(labels['True_labels'], labels['Defense'])
    conf_matrix_model = confusion_matrix(labels['True_labels'], labels['Original'])
    conf_matrix_adversarial = confusion_matrix(labels['True_labels'], labels['Adversarial'])

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=('Model', 'Adversarial', 'Defense'),
        horizontal_spacing=0.01
    )
    plot_sub_confusion_matrix(fig, conf_matrix_model, classes, 1)
    plot_sub_confusion_matrix(fig, conf_matrix_adversarial, classes, 2)
    plot_sub_confusion_matrix(fig, conf_matrix_defense, classes, 3, show_colorbar=True)

    fig.update_layout(title=dict(text="Confusion Matrices", x=0.5, xanchor='center'), height=600, width=1500)
    return fig




def generate_report_per_class(labels,classes,predictions,confidence):
    class_report = {}
    for idx, class_name in enumerate(classes):
        confidence_model = np.array([confidence[i] for i, label in enumerate(labels) if label == idx])
        positions = np.where(labels == idx)
        class_report[class_name] = {"accuracy":float(np.sum(labels[positions] == predictions[positions])/len(labels[positions])),
                                    "confidence_mean":float(confidence_model.mean()),
                                    "confidence_std":float(confidence_model.std()),
                                    "confidence_max":float(confidence_model.max()),
                                    "confidence_min":float(confidence_model.min()),
                                    "confidence_25":float(np.percentile(confidence_model,25)),
                                    "confidence_75":float(np.percentile(confidence_model,75))}
    return class_report
    

def generate_report(data_loader,model,data_params,get_original=False):
    model_classifier = init_model(model)
    model_classifier.eval()
    accuracy = 0
    top_k_accuracy = 0
    total = 0
    confidence = []
    confidence_top_k = []
    true_labels = []
    model_predictions = []
    model_prediction_top_k = []
    for batch in data_loader:
        if get_original:
            images, real_labels = batch["original"], batch["real_labels"]
        else:
            images, real_labels = batch["examples"], batch["real_labels"]
            
        images, real_labels = images.to(device), real_labels.to(device)
        
        with torch.no_grad():
            outputs = model_classifier(images)
            prediction_top_k = F.softmax(outputs,dim=1).topk(data_params["Top_k"], dim=-1)
            _, predicted = torch.max(outputs.data, 1)
            predicted_top_k = prediction_top_k[1].squeeze()
            confidence_top_k.extend(prediction_top_k[0].cpu().numpy())
            confidence.extend(F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy())
            total += real_labels.size(0)
            accuracy += (predicted == real_labels).sum().item()
            top_k_accuracy += (predicted_top_k == real_labels[...,None]).any(dim=-1).sum().item()
            true_labels.extend(real_labels.cpu().numpy())
            model_predictions.extend(predicted.cpu().numpy())
            model_prediction_top_k.extend(predicted_top_k.cpu().numpy())
    accuracy = float(accuracy) * 100 / total
    top_k_accuracy = float(top_k_accuracy) * 100 / total
    #get the report per class
    confidence = np.array(confidence)
    confidence_top_k = np.array(confidence_top_k)
    true_labels = np.array(true_labels)
    model_predictions = np.array(model_predictions)
    model_prediction_top_k = np.array(model_prediction_top_k)
    
    class_report = generate_report_per_class(true_labels,data_params["Classes_names"],model_predictions,confidence)
    
    elements_to_return = [accuracy, top_k_accuracy,
                          confidence,confidence_top_k,
                          true_labels.tolist(),model_predictions.tolist(),
                          class_report
                          ]
    return elements_to_return





def Generate_total_report(adversarial_data,data_params,model,transformations):
    #get the original report
    data_loader = General_defense_transform(adversarial_data,[],data_params)
    original_report = generate_report(data_loader,model,data_params,get_original=True)
    #get the adversarial report
    data_loader = General_defense_transform(adversarial_data,[],data_params)
    adversarial_report = generate_report(data_loader,model,data_params)
    #get the defense report
    data_loader = General_defense_transform(adversarial_data,transformations,data_params)
    defense_report = generate_report(data_loader,model,data_params)
    Accuracies = {"Original":original_report[0],
                    "Adversarial":adversarial_report[0],
                    "Defense":defense_report[0]}
    Top_k_Accuracies = {"Original":original_report[1],
                    "Adversarial":adversarial_report[1],
                    "Defense":defense_report[1]}
    Confidences = {"Original":original_report[2],
                    "Adversarial":adversarial_report[2],
                    "Defense":defense_report[2]}
    Confidences_top_k = {"Original":original_report[3],
                    "Adversarial":adversarial_report[3],
                    "Defense":defense_report[3]}
    Labels = {"Original":original_report[5],
              "True_labels":original_report[4],
              "Adversarial":adversarial_report[5],
              "Defense":defense_report[5]}
    Class_reports = {"Original":original_report[6],
                    "Adversarial":adversarial_report[6],
                    "Defense":defense_report[6]}
    To_Return={"Accuracies":Accuracies,
               "Top_k_Accuracies":Top_k_Accuracies,
              "Confidences":Confidences,
              "Confidences_top_k":Confidences_top_k,
              "Labels":Labels,
              "Class_reports":Class_reports}
    return To_Return





def run_single_defense(adversarial_data,model,transformations,data_params):
    report = Generate_total_report(adversarial_data,data_params,model,transformations)
    figs, Kl_dists = plot_confidence_distribution_all(report["Confidences"],report["Labels"],data_params["Classes_names"]) 
    conf_matrix = plot_confusion_matrix(labels=report["Labels"],classes=data_params["Classes_names"])
    keys_to_exclude = ["Confidences","Confidences_top_k","Labels"]
    for key in keys_to_exclude:
        report.pop(key,None)
    return report,figs,Kl_dists,conf_matrix



def run_all_defenses(adversarial_data,model,data_params)->Tuple[Dict[str,Dict[str,List]],Dict,Dict,Dict]:
    reports = {}
    figs = {}
    Kl_dists = {}
    conf_matrix = {}
    # transformations = {"Flip":FlipTransform,"JPEG":JPEGTransform}
    # transformations = {"Flip":FlipTransform,"ResizePad":ResizePadTransform,"Distort":DistortTransform,"JPEG":JPEGTransform}
    transformations = {"ResizePadFlip":ResizePadFlipTransform,"Distort":DistortTransform,"JPEG":JPEGTransform}
    logger = logging.getLogger(__name__)
    for i in range(1,len(transformations.keys())+1):
        for trans in list(itertools.permutations(transformations.keys(),i)):

            transformation_name = "_".join(trans)
            logger.info(f"Performing defense: {transformation_name}")
            to_apply = [transformations[name]() for name in trans]
            defense_results = run_single_defense(adversarial_data,model,to_apply,data_params)
            logger.info(f"{defense_results[0]['Accuracies']}")
            reports[transformation_name]= defense_results[0]
            figs[transformation_name] = defense_results[1]
            Kl_dists[transformation_name] = defense_results[2]
            conf_matrix[transformation_name] = defense_results[3]
    return reports,figs,Kl_dists,conf_matrix 


# def Report(dataloader:torch.utils.data.DataLoader,model:nn.Module,Report_params:Dict) -> Tuple[Dict,Dict,go.Figure,List[matplt.Figure]]:
#     model_classifier = init_model(model)
#     model_classifier.eval()
#     correct_defense = 0
#     correct_model = 0
#     correct_adversarial = 0
#     total = 0
#     confidence_defense, confidence, confidence_adversarial = [], [], []
#     true_labels, model_predictions, adversarial_predictions, defense_predictions = [], [], [], []
    
#     for batch in dataloader:
#         images, real_labels = batch["examples"], batch["real_labels"]
#         model_labels, adversarial_labels = batch["model_labels"], batch["adversarial_labels"]
#         model_confidence, adversarial_confidence = batch ["confidence"], batch["adversarial_confidence"]

#         images, real_labels = images.to(device), real_labels.to(device)
#         model_labels, adversarial_labels = model_labels.to(device), adversarial_labels.to(device)
#         model_confidence, adversarial_confidence = model_confidence.to(device), adversarial_confidence.to(device) 
#         with torch.no_grad():
#             outputs = model_classifier(images)
#             _, predicted = torch.max(outputs.data, 1)
#             predicted_top_k = outputs.topk(Report_params['Top_k'], dim=-1)[1].squeeze()
#             confidence_defense.extend(F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy())
#             confidence.extend(model_confidence.cpu().numpy())
#             confidence_adversarial.extend(adversarial_confidence.cpu().numpy())
#             total += real_labels.size(0)
#             correct_defense += (predicted == real_labels).sum().item()
#             correct_model += (model_labels == real_labels).sum().item()
#             correct_adversarial += (adversarial_labels == real_labels).sum().item()
#             true_labels.extend(real_labels.cpu().numpy())
#             model_predictions.extend(model_labels.cpu().numpy())
#             adversarial_predictions.extend(adversarial_labels.cpu().numpy())
#             defense_predictions.extend(predicted.cpu().numpy())
            

#     original_accuracy = correct_model *100 /total
#     adversarial_accuracy = correct_adversarial *100 /total
#     defense_accuracy = correct_defense *100 /total

#     Accuracies = {"Accuracy":original_accuracy,
#                   "Adversarial_accuracy":adversarial_accuracy, 
#                   "Defense_accuracy":defense_accuracy}
    
#     Labels = {"true_labels": true_labels, 
#               "model_labels": model_predictions, 
#               "adversarial_labels":adversarial_predictions, 
#               "defense_labels":defense_predictions}
    
#     Confidences = {"model_confidence":confidence,
#                    "adversarial_confidence":confidence_adversarial,
#                    "defense_confidence":confidence_defense}
    
#     Classes = Report_params["Classes"]
#     figs, Kl_dists = plot_confidence_distribution_all(Confidences,Labels,Classes) 
#     conf_matrix = plot_confusion_matrix(labels=Labels,classes=Classes)
    
#     return Accuracies,Kl_dists,conf_matrix ,figs
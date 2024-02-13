"""
This is a boilerplate pipeline 'adversarial_examples'
generated using Kedro 0.19.1
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import art
from art.estimators.classification import PyTorchClassifier
from typing import Tuple, Dict, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(model:torch.nn.Module,lr:float=0.001)->Tuple[nn.Module,nn.Module,optim.Optimizer]:
    # Move model to GPU if available
    model = model.to(device)
    if device == 'cuda':
        model= torch.nn.DataParallel(model)
        cudnn.benchmark = True
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return model, criterion, optimizer
def denormalize(tensor: torch.Tensor, mean= torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32), std =  torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)):
    denorm = torch.clone(tensor)
    for t, m, s in zip(denorm, mean, std):
        t.mul_(s).add_(m)
    return denorm


def normalize(tensor: torch.Tensor, mean= torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32), std =  torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)):
    norm = torch.clone(tensor)
    for t, m, s in zip(norm, mean, std):
        t.sub_(m).div_(s)
    return norm

def classification(model:nn.Module,parameters:Dict)-> art.estimators.classification.pytorch.PyTorchClassifier:
    model, criterion, optimizer = init_model(model)

    if device == "cuda":
        device_type = "gpu"
    else:
        device_type = "cpu"
    classes = parameters["classes"]
    input_shape = parameters["input_shape"]
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        nb_classes=classes,
        input_shape=input_shape,
        device_type=device_type,
    )
    return classifier

def Evasion_Attack(classifier:art.estimators.classification.pytorch.PyTorchClassifier,attack:Dict):
    attack_module = attack["module"]
    attack_type = attack["class"]
    attack_arguments = attack["kwargs"]
    
    attack_class=getattr(importlib.import_module(attack_module),attack_type)
    attack_instance = attack_class(classifier,**attack_arguments)


    return attack_instance

def Adversarial_generation(testloader:torch.utils.data.DataLoader,classifier: art.estimators.classification.pytorch.PyTorchClassifier, attack_params: Dict,test_data_information:Dict[str,List[float]])->Dict[str,torch.Tensor]:
    attack = Evasion_Attack(classifier,attack_params)
    logger = logging.getLogger(__name__)
    mean = test_data_information["mean"]
    std = test_data_information["std"]
    logger.info(f"Creating attack of type {type(attack)}")
    # testloader = Create_data_loader(batch_size=512)
    real_labels = []
    model_labels = []
    adversarial_examples = []
    adversarial_labels = []
    label_confidence = []
    adv_label_confidence = []
    for data in testloader:
        images, labels = data
        images_cpu = images.cpu().detach().numpy()
        x_test_adv = attack.generate(x=images_cpu,y=labels.cpu().numpy())
        with torch.no_grad():
            # perform inference
            label_prob = classifier.predict(images_cpu)
            adv_label_prob = classifier.predict(x_test_adv)
            
            # get confidence and labels for real data
            predictions = np.argmax(label_prob,axis=1)
            confidence_predictions = np.max(F.softmax(torch.tensor(label_prob),dim=1).cpu().numpy(),axis=1)
            
            # get confidence and labels for adversarial data
            adversarial_predictions = np.argmax(adv_label_prob,axis=1)
            confidence_predictions_adv = np.max(F.softmax(torch.tensor(adv_label_prob),dim=1).cpu().numpy(),axis=1)
            
        
        adversarial_denorm = [denormalize(torch.tensor(x),mean=mean, std = std) for x in x_test_adv]
        
        
        label_confidence.extend(confidence_predictions)
        adv_label_confidence.extend(confidence_predictions_adv)
        adversarial_examples.extend(adversarial_denorm)
        real_labels.extend(labels.cpu().numpy())
        model_labels.extend(predictions)
        adversarial_labels.extend(adversarial_predictions)
        
    all_adversarial_examples = torch.stack(adversarial_examples)
    all_real_labels = torch.tensor(real_labels)
    all_model_labels = torch.tensor(model_labels)
    all_adversarial_labels = torch.tensor(adversarial_labels)
    all_confidence_labels = torch.tensor(label_confidence)
    all_adversarial_confidence_labels = torch.tensor(adv_label_confidence)
    
    adversarial_data = {
        "examples": all_adversarial_examples,
        "confidence": all_confidence_labels,
        "adversarial_confidence": all_adversarial_confidence_labels,
        "real_labels": all_real_labels,
        "model_labels": all_model_labels,
        "adversarial_labels": all_adversarial_labels,
    }

    old_accuracy = (all_real_labels == all_model_labels).sum().item()/ all_real_labels.size(0)
    new_accuracy =  (all_real_labels == all_adversarial_labels).sum().item()/ all_real_labels.size(0)
    logger.info(f"Accuracy of the model was {old_accuracy*100:.2f}% and now is {new_accuracy*100:.2f}%")
    return adversarial_data
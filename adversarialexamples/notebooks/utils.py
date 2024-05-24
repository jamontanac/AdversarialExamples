
import torch
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
#implement a 2d FFT for an image using pytorch
def randomFFT(image,log=True):
    """
    Apply 2D FFT to a given image.
    img: a tensor of shape (C, H, W)
    """
    # Convert the image to grayscale
    gray_image = torch.mean(image, dim=0, keepdim=True)
    
    # Apply 2D FFT to the grayscale image
    fft = torch.fft.fft2(gray_image, norm="ortho")
    fft = torch.fft.fftshift(fft)
    fft = torch.norm(fft,dim=0) 
    if log:
        fft = torch.log(fft+1e-9)
    
    return fft

def plot_FFT(avg_diff,std_diff,size=(10,5),cmap = 'magma', title = ''):
    fig, ax = plt.subplots(1, 2, figsize=size)
    # Plot the average difference
    cbar1 = ax[0].imshow(20*avg_diff, cmap=cmap, interpolation='nearest')
    fig.colorbar(cbar1, ax=ax[0],shrink=0.7)
    ax[0].axis('off')
    ax[0].set_title(f"Average")

    # Plot the std
    cbar2 = ax[1].imshow(20*std_diff, cmap=cmap, interpolation='nearest')
    # cbar2 = ax[1].imshow(instance, cmap=cmap, interpolation='nearest')
    fig.colorbar(cbar2, ax=ax[1],shrink=0.7)
    ax[1].axis('off')
    ax[1].set_title(f"Std")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
def compute_eigen_vector(image):
    """
    Compute the eigen vectors of the image and its attack.
    """
    # Convert the image to grayscale
    gray_image = torch.mean(image, dim=0, keepdim=True).squeeze()
    dimensions = gray_image.shape
    gray_image = gray_image@gray_image.T
    eigen_values_image,eigen_vectors_image  = torch.linalg.eig(gray_image)
    

    return eigen_vectors_image
def compute_eigen_values(image,normalize = False):
    """
    Compute the eigen values of the image and its attack.
    """
    # Convert the image to grayscale
    gray_image = torch.mean(image, dim=0, keepdim=True).squeeze()
    dimensions = gray_image.shape
    gray_image = gray_image@gray_image.T
    eigen_values_image  = torch.linalg.eigvals(gray_image)
    if normalize:
        eigen_values_image = eigen_values_image/dimensions[0]

    return eigen_values_image
    # return eigen_values_image, eigen_values_adv
def plot_eigen_values(eigen_values, title = ''):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plot eigen values of noise
    axs[0].hist(eigen_values['eigen_values_noise'], bins=100, density=True)
    axs[0].set_yscale('log')
    axs[0].set_title('Eigen Values')

    # Plot eigen values of the fft of the noise
    axs[1].hist(eigen_values['eigen_values_noise_fft'], bins=100, density=True)
    axs[1].set_yscale('log')
    axs[1].set_title('FFT eigenvalues')

    plt.show()


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# def plot_FFT_3d(avg_diff,std_diff,title = '', display = False,dataset = 'CIFAR'):
#     x = np.arange(avg_diff.shape[1])
#     y = np.arange(avg_diff.shape[0])
#     # Create a figure with two subplots
#     fig = make_subplots(
#         rows=1, cols=2,
#         specs=[[{'type': 'surface'}, {'type': 'surface'}]],
#         subplot_titles=('Average', 'Std'),
#         horizontal_spacing=0.1,
#         vertical_spacing=0.1
#     )

#     # Add first surface plot
#     fig.add_trace(
#         go.Surface(z=avg_diff, x=x, y=y, colorscale='Cividis',colorbar=dict(title='Z', x=0.42)),
#         row=1, col=1
#     )

#     # Add second surface plot
#     fig.add_trace(
#         go.Surface(z=std_diff, x=x, y=y, colorscale='Cividis',colorbar=dict(title='Z2', x=1.02)),
#         row=1, col=2
#     )

#     # camera1 = dict(
#     #     eye=cameraone
#     # )
#     # camera2 = dict(
#     #     eye=cameratwo
#     # )
    
#     # Update layout, labels, and titles
#     fig.update_layout(
#         width=1200,  # Increase figure width
#         height=600,  # Increase figure height
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#             # camera=camera1
#         ),
#         scene2=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z2',
#             # camera=camera2
#         ),
#         autosize=True
#     )
#     if display:
#         fig.show()
#     else:
#         output_dir = f"Animations/FFT/{dataset}/"
#         ensure_dir(output_dir)
#         safe_title = title.replace('/','_').replace("\\", "_")
#         pio.write_image(fig, os.join(output_dir, f"{safe_title}.png"))



# # def plot_eigen_vectors(avg_eigen_vectors,std_eigen_vectors,title = '',dataset="intel",display=False, cameraone= dict(x=2.25, y=1.25, z=0.25), cameratwo = dict(x=1.25, y=1.25, z=1.25)):
# def plot_eigen_vectors(avg_eigen_vectors,std_eigen_vectors,title = '',dataset="intel",display=False):

#     avg_matrix = avg_eigen_vectors
#     std_matrix = std_eigen_vectors
#     log_avg_matrix = np.log1p(avg_matrix)
#     log_std_matrix = np.log1p(std_matrix)
#     x = np.arange(avg_matrix.shape[1])
#     y = np.arange(avg_matrix.shape[0])
#     # Create a figure with two subplots
#     fig = make_subplots(
#         rows=1, cols=2,
#         specs=[[{'type': 'surface'}, {'type': 'surface'}]],
#         subplot_titles=('avg eigenvectors', 'std_eigen_vectors'),
#         horizontal_spacing=0.1,
#         vertical_spacing=0.1
#     )

#     # Add first surface plot
#     fig.add_trace(
#         go.Surface(z=log_avg_matrix, x=x, y=y, colorscale='Cividis',colorbar=dict(title='Log(Z)', x=0.42)),
#         row=1, col=1
#     )

#     # Add second surface plot
#     fig.add_trace(
#         go.Surface(z=log_std_matrix, x=x, y=y, colorscale='Cividis',colorbar=dict(title='Log10(Z2)', x=1.02)),
#         row=1, col=2
#     )
#     # camera1 = dict(
#     #     eye=cameraone
#     # )
#     # camera2 = dict(
#     #     eye=cameratwo
#     # )
#     # Update layout, labels, and titles
#     fig.update_layout(
#         width=1200,  # Increase figure width
#         height=600,  # Increase figure height
#         scene=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Log(AVG)',
#             # camera=camera1
#         ),
#         scene2=dict(
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Log(STD)',
#             # camera=camera2
#         ),
#         autosize=True
#     )
#     if display:
#         fig.show()
#     else:
#         output_dir = f"Animations/eigen/{dataset}/"
#         ensure_dir(output_dir)
#         safe_title = title.replace('/','_').replace("\\", "_")
#         pio.write_image(fig, os.join(output_dir, f"{safe_title}.png"))

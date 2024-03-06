

import PIL
import io

import torchvision.transforms as transforms
import torch
 # Define the transformations to be applied to the adversarial examples   
def flip_image(image, axis=2):
    """
    Flips an image over a specific axis, default around y axis.

    Args:
        image (torch.Tensor): The input image to flip.

    Returns:
        torch.Tensor: The flipped image.
    """
    return image.flip(axis)

def resize_pad(image, ratio=0.9):
    """
    Resizes and pads an image with zeros to match the original size.

    Args:
        image (torch.Tensor): The input image to resize and pad.
        ratio (float): The ratio to resize the image by (default 0.8).

    Returns:
        torch.Tensor: The resized and padded image.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(image.shape[1] * ratio)),
        transforms.ToTensor()
    ])
    resized_image = transform(image)
    max_dim = image.shape[1] - resized_image.shape[1]
    start_y = torch.randint(0, max_dim + 1, (1,))
    start_x = torch.randint(0, max_dim + 1, (1,))
    padded_image = torch.zeros_like(image)
    padded_image[:,start_y:start_y + resized_image.shape[1], start_x:start_x + resized_image.shape[1]] = resized_image

    return padded_image

def random_distortion(h, w, d, delta):
    """
    Returns distorted coordinates
    """
    nw = w // d
    nh = h // d
    distorted_coords = torch.zeros(nh+1, nw+1, 2)
    
    for m in range(nw+1):
        for n in range(nh+1):
            dx = (torch.rand(1) * 2 - 1) * delta  
            dy = (torch.rand(1) * 2 - 1) * delta 
            x = m * d + dx
            y = n * d + dy
            distorted_coords[n, m, 0] = x
            distorted_coords[n, m, 1] = y
            
    return distorted_coords
    
def image_distortion(img, d=4, delta=0.5):
    """
    Apply distortion to a given image.
    img: a tensor of shape (C, H, W)
    d: size of the grid
    delta: distortion limit
    """
    C, H, W = img.shape
    nw = W // d
    nh = H // d
    distorted_coords = random_distortion(H, W, d, delta)
    distorted_image = torch.zeros_like(img)
    
    for m in range(nw+1):
        for n in range(nh+1):
            src_x = m * d
            src_y = n * d
            dest_x = int(distorted_coords[n, m, 0].item())
            dest_y = int(distorted_coords[n, m, 1].item())
            for i in range(d+1):
                for j in range(d+1):
                    if src_y + j < H and src_x + i < W and dest_y + j < H and dest_x + i < W:
                        distorted_image[:, dest_y + j, dest_x + i] = img[:, src_y + j, src_x + i]
                        
    return distorted_image

def JPEG_compression(image, quality=90,subsampling=0,optimize=True):
    """
    Apply JPEG compression to a given image.
    img: a tensor of shape (C, H, W)
    quality: quality of the compression
    """
    # quality = np.random.randint(60, 100)
    image = transforms.ToPILImage()(image)
    buffer = io.BytesIO()
    image.save(buffer, format='jpeg', quality=quality,subsampling=subsampling,optimize=optimize)
    buffer.seek(0)
    img = PIL.Image.open(buffer)
    img = transforms.ToTensor()(img)
    return img

# def randomWEBP(image, quality=90):
#     """
#     Apply WEBP compression to a given image.
#     img: a tensor of shape (C, H, W)
#     quality: quality of the compression
#     """
#     buffer = io.BytesIO()
#     image.save(buffer, format='webp', quality=quality)
#     buffer.seek(0)
#     img = PIL.Image.open(buffer)
#     return img
class FlipTransform:
    def __init__(self, axis=2):
        self.axis = axis

    def __call__(self, image):
        return flip_image(image, axis=self.axis)
class ResizePadTransform:
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def __call__(self, image):
        return resize_pad(image, ratio=self.ratio)
class ResizePadFlipTransform:
    def __init__(self, ratio=0.8, axis=2):
        self.ratio = ratio
        self.axis = axis

    def __call__(self, image):
        image = resize_pad(image, ratio=self.ratio)
        return flip_image(image, axis=self.axis)
class DistortTransform:
    def __init__(self, d=4, delta=0.5):
        self.d = d
        self.delta = delta
        
    def __call__(self, img):
        return image_distortion(img, self.d, self.delta)
class JPEGTransform:
    def __init__(self, quality=90,subsampling=0,optimize=True):
        self.quality = quality
        self.subsampling = subsampling
        self.optimize = optimize


    def __call__(self, image):
        return JPEG_compression(image, quality=self.quality,subsampling=self.subsampling,optimize=self.optimize)
# class WEBPTransform:
#     def __init__(self, quality=90):
#         self.quality = quality

#     def __call__(self, image):
#         return randomWEBP(image, quality=self.quality)

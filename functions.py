import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import torch
import torchvision
import sys
import os
import shutil
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.morphology import convex_hull_image
from skimage.morphology import convex_hull_image

def prepare_model(verbose=False):
    
    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        if verbose:
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
    
    # Prepare model
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    return predictor
    


def make_mask(img, predictor, step = 500, visualize = False):
    
    # Generate query points for SAM
    queries = np.array(list(itertools.product(
        np.arange(step//2, img.shape[1], step),
        np.arange(step//2, img.shape[0], step), 
    )))
    
    # Apply SAM2
    predictor.set_image(img)
    candidates = []
    
    for point in queries:
        input_point = np.array([point])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
    
        # Some selection
        for mask in masks:

            if visualize:
                plt.figure()
                plt.subplot(121)
                plt.imshow(img)
                plt.scatter([input_point[0,0]], [input_point[0,1]])
                plt.subplot(122)
                plt.imshow(mask)
                plt.show()
    
            # Size
            mask_size_min = 100000
            mask_size_max = 1000000
            mask_size = np.sum(mask)
            if mask_size < mask_size_min or mask_size > mask_size_max:
                if visualize:
                    print('Rejected because of size.')
                break
        
            # Centeredness
            mask_cx_min = mask.shape[0]*0.25
            mask_cx_max = mask.shape[0]*0.75
            mask_cx = np.mean(np.where(mask)[0])
            if mask_cx < mask_cx_min or mask_cx > mask_cx_max:
                if visualize:
                    print('Rejected because of centeredness.')
                break
            
            # mask_cy_min = mask.shape[1]*0.25
            # mask_cy_max = mask.shape[1]*0.75
            # mask_cy = np.mean(np.where(mask)[1])
            # if mask_cy < mask_cy_min or mask_cy > mask_cy_max:
            #     if visualize:
            #         print('Rejected because of centeredness.')
            #     break

            # Outer dimensions
            mask_xrange_min = 500
            mask_xrange_max = 2000
            mask_xrange = max(np.where(mask)[0])-min(np.where(mask)[0])
            if mask_xrange < mask_xrange_min or mask_xrange > mask_xrange_max:
                if visualize:
                    print('Rejected because of outer dimensions.')
                break
            
            mask_yrange_min = 500
            mask_yrange_max = 1500
            mask_yrange = max(np.where(mask)[1])-min(np.where(mask)[1])
            if mask_yrange < mask_yrange_min and mask_yrange > mask_yrange_max:
                if visualize:
                    print('Rejected because of outer dimensions.')
                break

            # Greenness
            mask_green_min = 40
            mask_coords = np.array(np.where(mask)).transpose()
            mask_rgb = np.array([img[x, y] for x, y in  mask_coords])
            # mask_green = (mask_rgb[:,0] < mask_rgb[:,1]) & (mask_rgb[:,2] < mask_rgb[:,1]) 
            # mask_green = mask_rgb[:,0] < mask_rgb[:,1]
            mask_green = 2*mask_rgb[:,1] - mask_rgb[:,0] - mask_rgb[:,2]
            mask_greenness = np.mean(mask_green)
            if mask_greenness < mask_green_min:
                if visualize:
                    print(f'Rejected because of greenness: {mask_greenness}')
                break

            # Filledness
            fill_rate_min = 0.8
            hull = convex_hull_image(mask)
            fill_rate = np.sum(hull & mask.astype(bool))/np.sum(hull)
            if fill_rate < fill_rate_min:
                if visualize:
                    print(f'Rejected because of filledness: {fill_rate}')
                break

            # All test passed        
            candidate = {
                'mask': mask,
                'greenness': mask_greenness,
                'fill_rate': fill_rate
            }
            candidates.append(candidate)

    # print(len(candidates), 'candidates')

    # Best candidate has highest product of greenness and fill_rate
    if len(candidates) > 0:
        i_winner = np.argmax([(can['greenness']*can['fill_rate']) for can in candidates])
        # i_winner = np.argmax([can['fill_rate'] for can in candidates])
        mask = candidates[i_winner]['mask']
        if visualize:
            plt.figure()
            plt.subplot(131)
            plt.imshow(img)
            plt.subplot(132)
            plt.imshow(img)
            plt.scatter(queries[:,0], queries[:,1])
            plt.subplot(133)
            plt.imshow(mask)
            plt.show()
        return mask

    else:
        print('WARNING: No mask generated.')
        if visualize:
            plt.figure()
            plt.subplot(131)
            plt.imshow(img)
            plt.subplot(132)
            plt.imshow(img)
            plt.scatter(queries[:,0], queries[:,1])
            plt.subplot(133)
            plt.imshow(np.zeros(img.shape))
            plt.show()
        return None



def fit_ellipse(mask):
    # Foreground pixel coordinates
    y, x = np.nonzero(mask)

    if len(x) < 3:
        raise ValueError("Mask must contain at least 3 foreground pixels.")

    # Center of mass
    cx = x.mean()
    cy = y.mean()

    # Centered coordinates
    dx = x - cx
    dy = y - cy

    # Covariance / second moment matrix
    cov = np.cov(np.stack([dx, dy]))

    # Eigenvalues/eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort: largest eigenvalue = major axis
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Axis directions
    major_axis = eigenvectors[:, 0]
    minor_axis = eigenvectors[:, 1]

    # Orientation of major axis
    angle = np.arctan2(major_axis[1], major_axis[0])

    # Convert to degrees
    angle_deg = np.degrees(angle)

    # Ellipse semi-axis lengths
    # For a uniform filled ellipse:
    # variance = semi_axis^2 / 4
    semi_major = 2 * np.sqrt(eigenvalues[0])
    semi_minor = 2 * np.sqrt(eigenvalues[1])

    return {
        "center": (cx, cy),
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "angle_rad": angle,
        "angle_deg": angle_deg,
    }

def plot_mask_ellipse(mask, ellipse, ax):

    # plt.figure()

    # Mask
    plt.imshow(mask, origin="upper")
    
    # Ellipse and center
    theta = np.linspace(0, 2*np.pi, 200)
    a = ellipse["semi_major"]
    b = ellipse["semi_minor"]
    cx, cy = ellipse["center"]
    angle = ellipse["angle_rad"]
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    ellipse_xy = R @ np.vstack([
        a * np.cos(theta),
        b * np.sin(theta)
    ])
    ellipse_x = ellipse_xy[0] + cx
    ellipse_y = ellipse_xy[1] + cy
    plt.plot(ellipse_x, ellipse_y)
    plt.scatter(cx, cy, c="yellow")
    
    # Major axis
    x = [
        ellipse['center'][0] - ellipse['semi_major']*ellipse['major_axis'][0],
        ellipse['center'][0] + ellipse['semi_major']*ellipse['major_axis'][0]
    ]
    y = [
        ellipse['center'][1] - ellipse['semi_major']*ellipse['major_axis'][1],
        ellipse['center'][1] + ellipse['semi_major']*ellipse['major_axis'][1]
    ]
    plt.plot(x,y)
    
    # Minor axis
    x = [
        ellipse['center'][0] - ellipse['semi_minor']*ellipse['minor_axis'][0],
        ellipse['center'][0] + ellipse['semi_minor']*ellipse['minor_axis'][0]
    ]
    y = [
        ellipse['center'][1] - ellipse['semi_minor']*ellipse['minor_axis'][1],
        ellipse['center'][1] + ellipse['semi_minor']*ellipse['minor_axis'][1]
    ]
    plt.plot(x,y)
    
    # Zoom box
    margin = 50
    plt.ylim(max(np.where(mask)[0])+margin, min(np.where(mask)[0])-margin)
    plt.xlim(min(np.where(mask)[1])-margin, max(np.where(mask)[1])+margin)
    
    # plt.axis("equal")
    # plt.show()


# def plot_mask_ellipse(mask, ellipse, ax):

#     # plt.figure()

#     # Mask
#     plt.imshow(mask, origin="upper")
    
#     # Ellipse and center
#     theta = np.linspace(0, 2*np.pi, 200)
#     a = ellipse["semi_major"]
#     b = ellipse["semi_minor"]
#     cx, cy = ellipse["center"]
#     angle = ellipse["angle_rad"]
#     R = np.array([
#         [np.cos(angle), -np.sin(angle)],
#         [np.sin(angle),  np.cos(angle)]
#     ])
#     ellipse_xy = R @ np.vstack([
#         a * np.cos(theta),
#         b * np.sin(theta)
#     ])
#     ellipse_x = ellipse_xy[0] + cx
#     ellipse_y = ellipse_xy[1] + cy
#     plt.plot(ellipse_x, ellipse_y)
#     plt.scatter(cx, cy, c="yellow")
    
#     # Major axis
#     x = [
#         ellipse['center'][0] - ellipse['semi_major']*ellipse['major_axis'][0],
#         ellipse['center'][0] + ellipse['semi_major']*ellipse['major_axis'][0]
#     ]
#     y = [
#         ellipse['center'][1] - ellipse['semi_major']*ellipse['major_axis'][1],
#         ellipse['center'][1] + ellipse['semi_major']*ellipse['major_axis'][1]
#     ]
#     plt.plot(x,y)
    
#     # Minor axis
#     x = [
#         ellipse['center'][0] - ellipse['semi_minor']*ellipse['minor_axis'][0],
#         ellipse['center'][0] + ellipse['semi_minor']*ellipse['minor_axis'][0]
#     ]
#     y = [
#         ellipse['center'][1] - ellipse['semi_minor']*ellipse['minor_axis'][1],
#         ellipse['center'][1] + ellipse['semi_minor']*ellipse['minor_axis'][1]
#     ]
#     plt.plot(x,y)
    
#     # Zoom box
#     margin = 50
#     plt.ylim(max(np.where(mask)[0])+margin, min(np.where(mask)[0])-margin)
#     plt.xlim(min(np.where(mask)[1])-margin, max(np.where(mask)[1])+margin)
    
#     # plt.axis("equal")
#     # plt.show()


def find_grid_spacing(img, visualize=False):
    
    # Blur
    blur = cv2.GaussianBlur(img, (15, 15), 0)  
    
    # Convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    if visualize:
        plt.figure(figsize=(20,10))
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(blur)
        plt.subplot(133)
        plt.imshow(gray)
        plt.show()
    
    # Horizontal edges
    # Gradient in Y direction
    horizontal = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Vertical edges
    # Gradient in X direction
    vertical = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Convert to displayable images
    horizontal = cv2.convertScaleAbs(horizontal)
    vertical = cv2.convertScaleAbs(vertical)
    horiverti = horizontal.astype(np.bool) & vertical.astype(np.bool)
    
    if visualize:
        plt.figure(figsize=(20,10))
        plt.subplot(131)
        plt.imshow(horizontal)
        plt.subplot(132)
        plt.imshow(vertical)
        plt.subplot(133)
        plt.imshow(horiverti)
        plt.show()
    
    # Blur a few times
    horiverti_blur = horiverti.copy()
    for i in range(3):
        horiverti_blur = cv2.GaussianBlur(horiverti_blur.astype(np.uint8)*255, (15, 15), 0)  
    
    # Thresh
    horiverti_threshold = horiverti_blur > 200
    
    # Connected components
    analysis = cv2.connectedComponentsWithStats(horiverti_threshold.astype(np.uint8), 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    
    # Mask size filter
    mask_size_min = 500
    mask_size_max = 5000
    
    centroids_filtered = []
    label_ids_filtered = label_ids.copy()
    for i in range(totalLabels):
        mask_size = np.sum(label_ids == i)
        if mask_size >= mask_size_min and mask_size <= mask_size_max:
            centroids_filtered.append(centroid[i])
            pos_drop = np.array(np.where(label_ids_filtered == i)).transpose()
            for x, y in pos_drop:
                label_ids_filtered[x,y] = 0    
    centroids_filtered = np.array(centroids_filtered)
    
    if visualize:
        print(len(centroid), len(centroids_filtered))
        plt.figure(figsize=(20,10))
        plt.subplot(131)
        plt.imshow(horiverti_threshold)
        plt.subplot(132)
        plt.imshow(label_ids)
        plt.scatter([c[0] for c in centroid], [c[1] for c in centroid], s = 100, marker='o',  facecolors='none', edgecolors='red')
        plt.subplot(133)
        plt.imshow(label_ids_filtered)
        plt.scatter([c[0] for c in centroids_filtered], [c[1] for c in centroids_filtered],
                    s = 100, marker='o',  facecolors='none', edgecolors='red')
        plt.show()
    
    # Distances to nearest neighbour
    detections = centroids_filtered
    distances = []
    for i in range(len(detections)):
        lowest_distance = np.inf
        for j in range(len(detections)):
            if not j == i:
                distance = np.linalg.norm(detections[i] - detections[j])
                lowest_distance = min(lowest_distance, distance)
        distances.append(lowest_distance)
    
    # Keep middle section
    distances.sort()
    distances = distances[len(distances)//5:4*len(distances)//5]
    
    # Grid spacing is the mean 
    grid_spacing = np.mean(distances)
    
    return grid_spacing


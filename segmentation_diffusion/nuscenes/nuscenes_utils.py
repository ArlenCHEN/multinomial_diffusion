from io import BytesIO
import cv2
import numpy as np
from shapely import affinity

NUSC_LIDAR_CLASS_NAMES = ['driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'car', 'truck',
                          'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle',
                          'bicycle', 'traffic_cone', 'barrier']

def transform(matrix, vectors):
    vectors = np.dot(matrix[:-1, :-1], vectors.T)
    vectors = vectors.T + matrix[:-1, -1]
    return vectors


def project_to_image_plane(intrinsics, cam_points):
    intrinsics_tfm = np.zeros((3, 4))
    intrinsics_tfm[:, :3] = intrinsics

    # make homogeneous coordinates
    cam_points_homogeneous = np.vstack([cam_points.T, np.ones((cam_points.shape[0],))])

    image_points_homogeneous = intrinsics_tfm @ cam_points_homogeneous
    image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]
    return image_points.T


def projection_bev_matrix(rx=4, rz=-4, ux=100, uz=200):
    """
    rx, rz: grid resolution: number of grid cells per meter in x and z direction
    ux, uz: coordinates of origin of the bev grid
    :return: M - bev_projection_matrix such that M @ np.array([x,z,1]).reshape(3,1) gives the grid cell coordinates

    for rx=4, rz=-4, ux=100, uz=200
    # (x, z) -> (bu, bv)
    # (-25, 0) -> (0, 200)
    # (25, 0) -> (200, 200)
    # (25, 50) -> (200, 0)
    # (-25, 50) -> (0, 0)
    """

    bev_projection_matrix = np.array([[rx, 0, ux],
                                      [0, rz, uz],
                                      [0, 0, 1]])
    return bev_projection_matrix


def transform_polygon(polygon, affine):
    """
    Transform a 2D polygon
    """
    a, b, tx, c, d, ty = affine.flatten()[:6]
    return affinity.affine_transform(polygon, [a, b, c, d, tx, ty])


def render_polygon(mask, polygon, extents, resolution, value=1):
    if len(polygon) == 0:
        return
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


def get_visible_mask(instrinsics, image_width, extents, resolution):
    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)


def get_occlusion_mask(points, extents, resolution):
    x1, z1, x2, z2 = extents

    # A 'ray' is defined by the ratio between x and z coordinates
    ray_width = resolution / z2
    ray_offset = x1 / ray_width
    max_rays = int((x2 - x1) / ray_width)

    # Group LiDAR points into bins
    rayid = np.round(points[:, 0] / points[:, 2] / ray_width - ray_offset)
    depth = points[:, 2]

    # Ignore rays which do not correspond to any grid cells in the BEV
    valid = (rayid > 0) & (rayid < max_rays) & (depth > 0)
    rayid = rayid[valid]
    depth = depth[valid]

    # Find the LiDAR point with maximum depth within each bin
    max_depth = np.zeros((max_rays,))
    np.maximum.at(max_depth, rayid.astype(np.int32), depth)

    # For each bev grid point, sample the max depth along the corresponding ray
    x = np.arange(x1, x2, resolution)
    z = np.arange(z1, z2, resolution)[:, None]
    grid_rayid = np.round(x / z / ray_width - ray_offset).astype(np.int32)
    grid_max_depth = max_depth[grid_rayid]

    # A grid position is considered occluded if the there are no LiDAR points
    # passing through it
    occluded = grid_max_depth < z
    return occluded


def make_composite(cls_maps):
    """
    cls_maps: Nc x h x w - boolean masks tensors
    Nc - number of classes - len(args.pred_classes_nusc)

    Output: h x w tensors where each pixel will have a class index
    """
    nc = cls_maps.shape[0]
    class_idx = np.arange(nc) + 1
    x = (cls_maps > 0.5).astype('float') * class_idx.reshape(-1, 1, 1)
    cls_map_composite = np.max(x, axis=0)
    
    cls_map_composite[cls_map_composite==2] = 1 # Label other_flat as drivable surface
    
    return cls_map_composite


def color_components(labels, color_map):
    """
    label 0 is assigned white color to have a white background.

    Iterates through the image to replace each pixel with the color associated with its label.

    Returns the colored image.
    """    
    if labels.shape[1] == 1:
        labels = np.squeeze(labels, axis=1) # Remove the channel axis    
    if len(labels.shape) == 2: # batch size is 1
        colors = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        for idx in np.unique(labels).astype(int):
            ii, jj = np.where(labels == idx)
            colors[ii, jj, :] = np.array(color_map[idx])
    elif len(labels.shape) == 3: # batch size is not 1
        colors = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype=np.uint8)
        for idx in np.unique(labels).astype(int):
            ii, jj, kk = np.where(labels == idx)
            colors[ii, jj, kk, :] = np.array(color_map[idx])
    return colors


def compute_iou(pred, labels):
    """
    returns iou_per_class and iou_per_sample

    iou_per_sample is average of ious of classes that are present in labels

    :param pred: (nc,h,w)
    :param labels: (nc,h,w)
    :return: (nc,), float64
    """
    intersection = pred * labels
    union = pred + labels
    iou_per_class = intersection.sum(axis=-1).sum(axis=-1) / (union.sum(axis=-1).sum(axis=-1) + 1e-5)

    class_counts = labels.sum(axis=-1).sum(axis=-1)
    class_counts[class_counts > 0] = 1
    num_classes_present = np.sum(class_counts)
    iou_per_sample = np.sum(iou_per_class) / num_classes_present

    return iou_per_class, iou_per_sample


def transform_pred_and_labels(bev_seg_map, bev_gt_map):
    """
    # remove other_flat (bev_seg_map - 2nd), ped_crossing - (bev_gt_map - 2nd)
    # remove terrain (bev_seg_map - 4th), carpark - (bev_gt_map - 4th)


    :param bev_seg_map: (14, 200, 200)
    :param bev_gt_map: (15, 196, 200)
    :return: (12, 196, 200), (12, 196, 200)
    """
    

    pred = transform_pred(bev_seg_map)
    labels = transform_label(bev_gt_map)

    return pred, labels

def transform_pred(bev_seg_map):
    num_classes = 14
    class_filter = np.ones((num_classes,))
    class_filter[1] = 0
    class_filter[3] = 0
    class_idxs = np.where(class_filter)[0]

    # reshape and convert to bool
    pred = bev_seg_map[class_idxs, :196, :]
    pred = pred > 0.5
    return pred

def transform_label(bev_gt_map):
    num_classes = 14
    class_filter = np.ones((num_classes,))
    class_filter[1] = 0
    class_filter[3] = 0
    class_idxs = np.where(class_filter)[0]

    # flip around and apply visible mask to ground truths
    labels = np.flip(bev_gt_map, 1)
    mask = labels[-1]
    labels = labels[class_idxs] * ~mask
    return labels


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)
import torch
import numpy as np


class Normalize(object):
    def __call__(self, verts):
        normalized_points = verts - np.mean(verts, axis=0)
        max_norm = np.max(np.linalg.norm(normalized_points, axis=1))

        normalized_points = normalized_points / max_norm

        return normalized_points


class RandomRotate(object):
    def __call__(self, verts):
        theta = 2 * np.random.uniform() * np.pi
        rotation_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        rotated = np.matmul(verts, rotation_mat)

        return rotated


class RandomNoise(object):
    def __call__(self, verts):
        noise = np.random.normal(0, 0.01, verts.shape)
        noise = np.clip(noise, -0.05, 0.05)
        return verts + noise


class ToTensor(object):
    def __call__(self, verts):
        return torch.from_numpy(verts)


class Translate(object):
    def __call__(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
          
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud
    

class Shuffle(object):
    def __call__(self, verts):
        np.random.shuffle(verts)
        return verts
    

class RandomShift(object):
    def __call__(self, verts, shift_range=0.1):
        shift = np.random.uniform(-shift_range, shift_range, (1, verts.shape[1]))
        return verts + shift
    

class RandomRotatePerturbation(object):
    def __call__(self, verts, angle_sigma=0.06, angle_clip=0.18):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        
        R = np.dot(Rz, np.dot(Ry, Rx))
        rotated = np.matmul(verts, R)

        return rotated
    

class RandomScale(object):
    def __call__(self, verts, low=0.8, high=1.25):
        scale = np.random.uniform(low, high)
        scaled = verts * scale
        return scaled
    

class RandomDropPoint(object):
    def __call__(self, pc, max_dropout_ratio=0.875):
        
        dropout_ratio = np.random.random() * max_dropout_ratio 
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

        if len(drop_idx) > 0:
            pc[drop_idx,:] = pc[0,:] 
        
        return pc
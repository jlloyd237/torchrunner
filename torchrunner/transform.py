import numpy as np
import cv2
import PIL


def generate_shadow_vertices(imshape, n_shadows=1, max_vertices=10):
    vertices_list = []
    for i in range(n_shadows):
        vertices=[]
        for d in range(np.random.randint(3, max_vertices)):  # num of vertices in the shadow polygon
            # vertices.append((imshape[1] * np.random.uniform(), imshape[0] // 3 + imshape[0] * np.random.uniform()))
            vertices.append((imshape[1] * np.random.uniform(), imshape[0] * np.random.uniform()))
        vertices = np.array([vertices], dtype=np.int32)
        vertices_list.append(vertices)
    return vertices_list

def add_shadow(image, n_shadows=1, shadow_coeff=0.75):
    mask = np.zeros_like(image)
    vertices_list = generate_shadow_vertices(image.shape, n_shadows)
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)
    image = image.copy()
    image[mask==255] = image[mask==255] * shadow_coeff
    return image


class Shadow:
    def __init__(self, n_shadows=1, shadow_coeff=0.75):
        self.n_shadows = n_shadows
        self.shadow_factor = shadow_coeff

    def __call__(self, image):
        image = np.asarray(image)
        image = add_shadow(image, self.n_shadows, self.shadow_factor)
        image = PIL.Image.fromarray(image)
        return image


class Crop:
    def __init__(self, box=None):
        self.box = box  # box = (left, top, right, bottom)

    def __call__(self, image):
        return image.crop(self.box)


class AdaptiveThreshold:
    def __init__(self, block_size=11, c=-5):
        self.block_size = block_size
        self.c = c

    def __call__(self, image):
        #         image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        image = np.asarray(image)
        #         _, image = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size,
                                      self.c)
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        return image

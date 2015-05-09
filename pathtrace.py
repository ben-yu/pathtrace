from math import sqrt
from random import uniform
import sys


class Vector:
    """
    3D Vector
    """
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, b):
        return Vector(self.x + b.x, self.y + b.y, self.z + b.z)

    def __sub__(self, b):
        return Vector(self.x - b.x, self.y - b.y, self.z - b.z)

    def __mul__(self, b):
        if isinstance(b, (int, float)):
            return Vector(self.x * b, self.y * b, self.z * b)
        return Vector(self.x * b.x, self.y * b.y, self.z * b.z)

    def norm(self):
        self = self * (1 / sqrt(self.x * self.x + self.y * self.y + self.z * self.z))
        return self

    def dot(self, b):
        return self.x * b.x + self.y * b.y + self.z * b.z

    def __mod__(self, b):
        return Vector(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x
        )


class Scene:
    def __init__(self, width=1024, height=768, fov=.5135):
        self.spheres = (
            Sphere(1e5, Vector(1e5 + 1, 40.8, 81.6), Vector(), Vector(.75, .25, .25), 'DIFF'),  # Left
            Sphere(1e5, Vector(-1e5 + 99, 40.8, 81.6), Vector(), Vector(.25, .25, .75), 'DIFF'),  # Right
            Sphere(1e5, Vector(50, 40.8, 1e5), Vector(), Vector(.75, .75, .75), 'DIFF'),  # Back
            Sphere(1e5, Vector(50, 40.8, -1e5 + 170), Vector(), Vector(), 'DIFF'),  # Front
            Sphere(1e5, Vector(50, 1e5, 81.6), Vector(), Vector(.75, .75, .75), 'DIFF'),  # Bottom
            Sphere(1e5, Vector(50, -1e5 + 81.6, 81.6), Vector(), Vector(.75, .75, .75), 'DIFF'),  # Top
            Sphere(16.5, Vector(27, 16.5, 47), Vector(), Vector(1., 1., 1.) * 999, 'SPEC'),  # Mirror
            Sphere(16.5, Vector(73, 16.5, 78), Vector(), Vector(1., 1., 1.) * 999, 'REFR'),  # Glass
            Sphere(600, Vector(50, 681.6 - .27, 81.6), Vector(12, 12, 12), Vector(), 'DIFF'),  # Lite
        )

        self.width = width
        self.height = height
        self.fov = fov

        self.img = [0] * (width * height)

        self.camera = Ray(Vector(50, 52, 295.6), Vector(0, -0.042612, -1).norm())
        self.c_x = Vector(self.width * self.fov / self.height)  # Horizontal camera direction
        self.c_y = (self.c_x % self.camera.d).norm() * self.fov    # Vertical camera direction

    def intersect(self, ray):
        temp = 1e20
        for i, sphere in enumerate(self.spheres):
            d = sphere.intersect(ray)
            if d < temp:
                temp = d
                index = i

        return (temp, index) if temp < 1e20 else None

    def clamp(self, x):
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    def render(self, samples=1):
        for y in range(self.height):
            x_i = (0, 0, y ** 3)
            for x in range(self.width):
                # 2x2 Subsampling
                image_index = (self.height - y - 1) * self.width + x
                for sy in range(2):
                    for sx in range(2):
                        radiance_vec = Vector()
                        for s in range(samples):
                            # Tent filter
                            r1 = 2 * uniform(0., 1.)
                            r2 = 2 * uniform(0., 1.)
                            dx = sqrt(r1) - 1 if r1 < 1 else 1 - sqrt(2 - r1)
                            dy = sqrt(r2) - 1 if r2 < 1 else 1 - sqrt(2 - r2)
                            # Ray direction
                            d = (self.c_x * (((sx + .5 + dx) / 2 + x) / self.width - .5) +
                                 self.c_y * (((sy + .5 + dy) / 2 + y) / self.height - .5) +
                                 self.camera.d)
                            # Radiance
                            radiance_vec += (self.radiance(Ray(self.camera.o + d * 140, d.norm()), 0, x_i)
                                             * (1. / samples))
                    self.img[image_index] += Vector(self.clamp(radiance_vec.x),
                                                    self.clamp(radiance_vec.y),
                                                    self.clamp(radiance_vec.z)) * .25

        return self.img


class Ray:
    """
    Ray defines a parametric line, where a point can be defined
    as: P(t) = origin + t * direction
    """
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


class Sphere:

    RAY_EPSILON = 1e-4

    def __init__(self, radius, position):
        self.radius = radius
        self.position = position

    def intersect(self, ray):
        ray_sphere_dir = ray.origin - self.position
        b = ray_sphere_dir.dot(ray.direction)
        determinant = b * b - ray_sphere_dir.dot(ray_sphere_dir) + self.radius * self.radius

        if determinant < 0:
            return None

        determinant = sqrt(determinant)

        t = b - determinant
        if t > Sphere.RAY_EPSILON:
            return t

        t = b + determinant
        if t > Sphere.RAY_EPSILON:
            return t

        return None


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: {0} sample_per_pixel'.format(sys.argv[0])
        exit()
    samples = int(sys.argv[1]) / 4 if len(sys.argv) == 2 else 1
    scene = Scene()
    img = scene.render(samples)
    # write to file
    with open('sample.ppm', 'w') as f:
        f.write('P3\n%d %d\n%d\n' % (scene.width, scene.height, 255))
        for px in img:
            f.write('%d %d %d ' % (px.x, px.y, px.z))
    print 'Done.'

from math import sqrt, pi, sin, cos, fabs
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

        self.img = [0] * self.width * self.height

        self.camera = Ray(Vector(50, 52, 295.6), Vector(0, -0.042612, -1).norm())
        self.c_x = Vector(self.width * self.fov / self.height)  # Horizontal camera direction
        self.c_y = (self.c_x % self.camera.direction).norm() * self.fov    # Vertical camera direction

    def intersect(self, ray):
        temp = 1e20
        for i, sphere in enumerate(self.spheres):
            d = sphere.intersect(ray)
            if d and d < temp:
                temp = d
                index = i

        return (temp, index) if temp < 1e20 else None

    def clamp(self, x):
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    def to_color(self, x):
        return int(self.clamp(x) ** (1 / 2.2) * 255 + 0.5)

    def radiance(self, ray, depth, e=1.):
        if depth > 10:
            return Vector()

        hit = self.intersect(ray)
        if not hit:
            return Vector()     # return black

        t, sid = hit
        obj = self.spheres[sid]

        x = ray.origin + ray.direction * t  # ray-scene intersection point
        n = (x - obj.position).norm()  # sphere normal
        nl = n if n.dot(ray.direction) else -n
        f = obj.color  # sphere BRDF modulator
        p = f.x if (f.x > f.y and f.x > f.z) else (f.y if f.y > f.z else f.z)

        depth += 1
        if depth > 5 or not p:
            if uniform(0., 1.) < p:
                f = f * (1. / p)
            else:
                return obj.emission * e

        # ideal diffuse reflection
        if obj.reflection_type == 'DIFF':
            r1 = 2 * pi * uniform(0., 1.)  # sample an angle
            r2 = uniform(0., 1.)
            r2s = sqrt(r2)  # sample a distance from center
            # create a random orthonormal coordinate frame (w, u, v)
            w = nl
            u = ((Vector(0, 1., 0) if fabs(w.x) > .1 else Vector(1., 0, 0)) % w).norm()
            v = w % u
            d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm()  # a random reflection ray
            return obj.emission + f * self.radiance(Ray(x, d), depth)

        # # ideal specular reflection
        elif obj.reflection_type == 'SPEC':
            return obj.emission + f * self.radiance(Ray(x, ray.direction - n * 2 * n.dot(ray.direction)), depth)

        # ideal dielectric refraction
        refl_ray = Ray(x, ray.direction - n * 2 * n.dot(ray.direction))
        into = n.dot(nl) > 0  # if ray from outside in
        nc = 1
        nt = 1.5
        nnt = nc / nt if into else nt / nc
        ddn = ray.direction.dot(nl)
        cos2t = 1 - nnt * nnt * (1 - ddn * ddn)
        # total internal reflection
        if cos2t < 0:
            return obj.emission + f * self.radiance(refl_ray, depth)
        # choose reflection/refraction
        tdir = (ray.direction * nnt - n * ((1 if into else -1) * (ddn * nnt + sqrt(cos2t)))).norm()
        a = nt - nc
        b = nt + nc
        R0 = a * a / (b * b)
        c = 1 - (-ddn if into else tdir.dot(n))
        Re = R0 + (1 - R0) * c ** 4
        Tr = 1 - Re
        P = .25 + .5 * Re
        RP = Re / P
        TP = Tr / (1 - P)

        # russian roulette
        if depth > 2:
            if uniform(0., 1.) < P:
                return obj.emission + f * self.radiance(refl_ray, depth) * RP
            else:
                return obj.emission + f * self.radiance(Ray(x, tdir), depth) * TP
        else:
            return self.radiance(refl_ray, depth) * Re + self.radiance(Ray(x, tdir), depth) * Tr

    def render(self, samples=1):
        for y in range(self.height):
            print('Rendering ({0} spp) {1}%'.format(samples * 4, 100. * y / (self.height - 1)))
            for x in range(self.width):
                # 2x2 Subsampling
                image_index = (self.height - y - 1) * self.width + x
                self.img[image_index] = Vector()
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
                                 self.camera.direction)
                            # Radiance
                            radiance_vec = radiance_vec + (self.radiance(Ray(self.camera.origin + d * 140, d.norm()), 0)
                                                           * (1. / samples))
                    self.img[image_index] = self.img[image_index] + Vector(self.clamp(radiance_vec.x),
                                                                           self.clamp(radiance_vec.y),
                                                                           self.clamp(radiance_vec.z)) * .25
                image_index = image_index + 1

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

    def __init__(self, radius, position, emission, color, reflection_type):
        self.radius = radius
        self.position = position
        self.emission = emission
        self.color = color
        self.reflection_type = reflection_type

    def intersect(self, ray):
        ray_sphere_dir =  self.position - ray.origin
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
        print('Usage: {0} sample_per_pixel'.format(sys.argv[0]))
        exit()
    samples = int(int(sys.argv[1]) / 4) if len(sys.argv) == 2 else 1
    scene = Scene()
    img = scene.render(samples)

    # write to file
    with open('sample.ppm', 'w') as f:
        f.write('P3\n%d %d\n%d\n' % (scene.width, scene.height, 255))
        for px in img:
            f.write('%d %d %d ' % (scene.to_color(px.x), scene.to_color(px.y), scene.to_color(px.z)))
    print('Done.')

"""
RayTracing
"""


import numpy as np
import matplotlib.pyplot as plt

w = 400
h = 300


def normalize(x):
    #algebra linear
    x /= np.linalg.norm(x)
    return x


def intersect_plane( O, D, P, N):
    # Retorne a distância de O até a interseção do raio (O, D) com o
    # plano (P, N) ou + inf se não houver interseção.
    # O e P são pontos 3D, D e N (normais) são vetores normalizados.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_sphere(O, D, S, R):
    # Retorne a distância de O até a interseção do raio (O, D) com o
    # esfera (S, R) ou + inf se não houver interseção.
    # O e S são pontos 3D, D (direção) é um vetor normalizado, R é um escalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])


def get_normal(obj, M):
    # encontra o normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    return N


def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color


def trace_ray(rayO, rayD):
    # Encontre o primeiro ponto de intersecção com a cena.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Retorne Nenhum se o raio não interceptar nenhum objeto.
    if t == np.inf:
        return
    # encontra o objeto.
    obj = scene[obj_idx]
    # Encontre o ponto de intersecção no objeto.
    M = rayO + rayD * t
    # Encontre propriedades do objeto.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Sombra: descubra se o ponto está sombreado ou não.
    l = [intersect(M + N * .0001, toL, obj_sh)
         for k, obj_sh in enumerate(scene) if k != obj_idx]
    if l and min(l) < np.inf:
        return
    # começa a computara cor
    col_ray = ambient
    # Sombreamento
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, M, N, col_ray


def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color), reflection=.5)


def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (color_plane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=.25)


# lista de objetos.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
         add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
         add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
         ]

# posição da luz e da cor.
L = np.array([10., 5., -20.])
color_light = np.ones(3)

# Parâmetros padrão de luz e material.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 5  # Número máximo de reflexos de luz.
col = np.zeros(3)  # cor atual .
O = np.array([0., 0.35, -1.])  # Camera.
Q = np.array([0., 0., 0.])  # Câmera apontando para.
img = np.zeros((h, w, 3))

r = float(w) / h
# Coordenadas da tela: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop por todos os pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print
        i / float(w) * 100, "%"
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # loop pelos raios iniciais e secundários.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # Reflexão: crie um novo ray.
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('imageRayTracingRodrigo.png', img)

import ast, json, math, operator, re
from collections import namedtuple
import numpy as np
from . import cadastre
from .core import compile_expr
from .rpc import rpc

PRIOR_STATUS = 'supposé'
NAME = re.compile(r'[A-Za-z_]\w*')
COMPONENT_ID = re.compile(r'[A-Za-z][A-Za-z0-9_]*')
SCHEMA = 'create table if not exists components(id text primary key, type text, json text)'
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
OTHER_AXIS = {'x': 'z', 'z': 'x'}
AREA_TOLERANCE = 1e-3


class Expr(dict):
    @staticmethod
    def of(value):
        if isinstance(value, dict): return value if isinstance(value, Expr) else Expr(value)
        if isinstance(value, (int, float)): return Expr({'1': float(value)} if value else {})
        return parse_linear(ast.parse(str(value), mode='eval').body)

    def constant(self): return self.get('1', 0.0) if set(self) <= {'1'} else None

    def wrapped(self):
        text = str(self)
        return text if re.fullmatch(r'[\w.]+', text) else f'({text})'

    def __neg__(self): return Expr({term: -coef for term, coef in self.items()})

    def __add__(self, other):
        total = Expr(self)
        for term, coef in Expr.of(other).items():
            total[term] = total.get(term, 0.0) + coef
            if abs(total[term]) < 1e-12: del total[term]
        return total

    __radd__ = __add__
    def __sub__(self, other): return self + -Expr.of(other)
    def __rsub__(self, other): return Expr.of(other) + -self

    def __mul__(self, other):
        other = Expr.of(other)
        for expr, factor in ((self, other), (other, self)):
            if (k := factor.constant()) is not None: return Expr({term: coef * k for term, coef in expr.items() if coef * k})
        return Expr({f'{self.wrapped()} * {other.wrapped()}': 1.0})

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Expr.of(other); k = other.constant()
        return self * (1 / k) if k else Expr({f'{self.wrapped()} / {other.wrapped()}': 1.0})

    def __str__(self):
        text = ''
        for term, coef in self.items():
            size = abs(coef); body = f'{size:.12g}' if term == '1' else term if size == 1 else f'{size:.12g} * {term}'
            text += ((' - ' if coef < 0 else ' + ') if text else ('-' if coef < 0 else '')) + body
        return text or '0'


def parse_linear(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return Expr.of(node.value)
    if isinstance(node, ast.Name): return Expr({node.id: 1.0})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -parse_linear(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](parse_linear(node.left), parse_linear(node.right))
    text = ast.unparse(node)
    return Expr({text if isinstance(node, ast.Call) else f'({text})': 1.0})


def call(function, *args): return Expr({f'{function}({", ".join(str(Expr.of(a)) for a in args)})': 1.0})


def evaluate(expr, values):
    return compile_expr(str(Expr.of(expr)), {n: i for i, n in enumerate(values)})[0](np.array(list(values.values()), float))


def param_names(comp): return {n for expr in comp['expressions'].values() for n in NAME.findall(str(expr))}
def evaluated(comp, evaluate_expr): return {k: evaluate_expr(v) for k, v in comp['expressions'].items()}


def box_shape(center, size, rot_y=0.0, **extra):
    return {'kind': 'box', 'center': [float(v) for v in center], 'size': [abs(float(v)) for v in size], 'rotY': float(rot_y), **extra}


def build_wall(comp, quantity, initial, components):
    expr = comp['expressions']; side = float(comp['options'].get('sens', 1)); run_axis = comp['options'].get('axe')
    if run_axis in OTHER_AXIS:
        pos_axis = OTHER_AXIS[run_axis]
        position = quantity(pos_axis, pos_axis, f'position ({pos_axis})', 0.0)
        start, end = quantity(run_axis + '0', run_axis, f'début ({run_axis})', 0.0), quantity(run_axis + '1', run_axis, f'fin ({run_axis})', 4.0)
        expr.update({f'start_{run_axis}': start, f'end_{run_axis}': end, f'start_{pos_axis}': position, f'end_{pos_axis}': position,
                     f'dir_{run_axis}': Expr.of(1.0 if initial(end) >= initial(start) else -1.0), f'dir_{pos_axis}': Expr(),
                     f'normal_{run_axis}': Expr(), f'normal_{pos_axis}': Expr.of(side)})
    elif run_axis: raise ValueError("mur : axe = 'x', 'z' ou vide (en biais)")
    else:
        start_x, start_z = quantity('x0', 'x', 'début (x)', 0.0), quantity('z0', 'z', 'début (z)', 0.0)
        end_x, end_z = quantity('x1', 'x', 'fin (x)', 4.0), quantity('z1', 'z', 'fin (z)', 0.0)
        length = call('hypot', end_x - start_x, end_z - start_z); dir_x, dir_z = (end_x - start_x) / length, (end_z - start_z) / length
        expr.update(start_x=start_x, start_z=start_z, end_x=end_x, end_z=end_z, dir_x=dir_x, dir_z=dir_z, normal_x=-side * dir_z, normal_z=side * dir_x)
    quantity('ep', 'dim', 'épaisseur', 0.2); quantity('h', 'dim', 'hauteur', 2.5); quantity('y0', 'y', 'base', '0')


def wall_anchors(comp):
    expr = comp['expressions']
    def opposite(x, z): return {'x': x + expr['normal_x'] * expr['ep'], 'z': z + expr['normal_z'] * expr['ep']}
    anchors = {'a': {'x': expr['start_x'], 'z': expr['start_z']}, 'b': {'x': expr['end_x'], 'z': expr['end_z']},
               'a_ext': opposite(expr['start_x'], expr['start_z']), 'b_ext': opposite(expr['end_x'], expr['end_z']),
               'bas': {'y': expr['y0']}, 'haut': {'y': expr['y0'] + expr['h']}}
    pos_axis = OTHER_AXIS.get(comp['options'].get('axe'))
    if pos_axis: anchors.update(face={pos_axis: expr[f'start_{pos_axis}']}, face_ext={pos_axis: anchors['a_ext'][pos_axis]})
    return anchors


def wall_geometry(comp, evaluate_expr, walls):
    val = evaluated(comp, evaluate_expr); dx, dz, thickness = val['end_x'] - val['start_x'], val['end_z'] - val['start_z'], val['ep']
    center = [(val['start_x'] + val['end_x'] + val['normal_x'] * thickness) / 2, val['y0'] + val['h'] / 2, (val['start_z'] + val['end_z'] + val['normal_z'] * thickness) / 2]
    return [box_shape(center, [math.hypot(dx, dz), val['h'], thickness], math.atan2(-dz, dx))]


WALL_FRAME = ('start_x', 'start_z', 'dir_x', 'dir_z', 'normal_x', 'normal_z', 'ep', 'y0')
OPENING_DEFAULTS = {'porte': {'l': 0.83, 'h': 2.04, 'al': '0'}, 'fenetre': {'l': 1.2, 'h': 1.25, 'al': 0.9}, 'porte_fenetre': {'l': 1.4, 'h': 2.15, 'al': '0'}}


def build_opening(comp, quantity, initial, components):
    wall = components.get(comp['options'].get('mur'))
    if not wall or wall['type'] != 'mur': raise ValueError(f"{TYPES[comp['type']].label} : option « mur » = id d’un mur existant")
    comp['expressions'].update({k: wall['expressions'][k] for k in WALL_FRAME})
    defaults = OPENING_DEFAULTS[comp['type']]
    quantity('pos', 'dim', 'position depuis l’extrémité a du mur', 0.5)
    quantity('l', 'dim', 'largeur', defaults['l']); quantity('h', 'dim', 'hauteur', defaults['h']); quantity('al', 'dim', 'allège', defaults['al'])


def opening_anchors(comp):
    expr = comp['expressions']
    def along(t): return {'x': expr['start_x'] + t * expr['dir_x'], 'z': expr['start_z'] + t * expr['dir_z']}
    def across(point, fraction=1.0):
        return {'x': point['x'] + fraction * expr['normal_x'] * expr['ep'], 'z': point['z'] + fraction * expr['normal_z'] * expr['ep']}
    left, right, sill = along(expr['pos']), along(expr['pos'] + expr['l']), expr['y0'] + expr['al']
    return {'g': left, 'd': right, 'g_ext': across(left), 'd_ext': across(right), 'bas': {'y': sill}, 'haut': {'y': sill + expr['h']},
            'centre': {**across(along(expr['pos'] + 0.5 * expr['l']), 0.5), 'y': sill + 0.5 * expr['h']}}


def opening_geometry(comp, evaluate_expr, walls):
    val = evaluated(comp, evaluate_expr); center = opening_anchors(comp)['centre']
    return [box_shape([evaluate_expr(center[a]) for a in 'xyz'], [val['l'], val['h'], val['ep'] + 0.02], math.atan2(-val['dir_z'], val['dir_x']))]


def set_vertices(comp, vertices, make):
    if vertices and len(vertices) < 3: raise ValueError('polygone : au moins 3 sommets')
    for k, (x, z) in enumerate(vertices, 1): comp['expressions'].update({f'x{k}': make(k, 'x', x), f'z{k}': make(k, 'z', z)})
    comp['vertex_count'] = len(vertices)


def build_level(comp, quantity, initial, components):
    quantity('y', 'y', 'niveau', '0' if comp['type'] == 'sol' else 2.5)
    set_vertices(comp, comp['options'].get('contour') or [], lambda k, axis, value: Expr.of(value))


def build_parcel(comp, quantity, initial, components):
    quantity('y', 'y', 'niveau', '0')
    set_vertices(comp, comp['options'].get('sommets') or [[0, 0], [20, 0], [20, 15], [0, 15]],
                 lambda k, axis, value: quantity(f'{axis}{k}', axis, f'sommet {k} ({axis})', value))


def polygon_anchors(comp):
    expr = comp['expressions']
    return {'niveau': {'y': expr['y']}, **{f's{k}': {'x': expr[f'x{k}'], 'z': expr[f'z{k}']} for k in range(1, comp['vertex_count'] + 1)}}


def polygon_geometry(comp, evaluate_expr, walls):
    expr = comp['expressions']
    vertices = [[evaluate_expr(expr[f'x{k}']), evaluate_expr(expr[f'z{k}'])] for k in range(1, comp['vertex_count'] + 1)]
    if not vertices and walls:
        xs, zs = zip(*[(evaluate_expr(w['expressions'][f'{end}_x']), evaluate_expr(w['expressions'][f'{end}_z'])) for w in walls for end in ('start', 'end')])
        vertices = [[min(xs), min(zs)], [max(xs), min(zs)], [max(xs), max(zs)], [min(xs), max(zs)]]
    return [{'kind': 'polygon', 'vertices': vertices, 'y': evaluate_expr(expr['y'])}] if vertices else []


def build_stairs(comp, quantity, initial, components):
    options = comp['options']; run_axis = options.setdefault('axe', 'x')
    if run_axis not in OTHER_AXIS or int(options.setdefault('n', 13)) < 2: raise ValueError("escalier : axe = 'x' ou 'z', n ≥ 2")
    side_axis = OTHER_AXIS[run_axis]
    quantity(run_axis, run_axis, f'1re contremarche ({run_axis})', 0.0); quantity(side_axis, side_axis, f'côté ({side_axis})', 0.0)
    quantity('g', 'escalier', 'giron', 0.25); quantity('r', 'escalier', 'hauteur de marche', 0.175)
    quantity('l', 'dim', 'largeur', 0.9); quantity('y0', 'y', 'base', '0')


def stairs_layout(comp):
    options = comp['options']; run_axis = options['axe']
    return run_axis, OTHER_AXIS[run_axis], int(options['n']), float(options.get('sens', 1)), float(options.get('cote', 1))


def stairs_anchors(comp):
    expr = comp['expressions']; run_axis, side_axis, count, run_sign, side_sign = stairs_layout(comp)
    def riser(k): return {run_axis: expr[run_axis] + run_sign * (k - 1) * expr['g']}
    anchors = {'depart': riser(1), 'arrivee': riser(count), 'cote': {side_axis: expr[side_axis]},
               'cote_ext': {side_axis: expr[side_axis] + side_sign * expr['l']}, 'bas': {'y': expr['y0']}, 'haut': {'y': expr['y0'] + count * expr['r']}}
    for k in range(1, count + 1): anchors.update({f'contremarche_{k}': riser(k), f'marche_{k}': {'y': expr['y0'] + k * expr['r']}})
    return anchors


def stairs_geometry(comp, evaluate_expr, walls):
    val = evaluated(comp, evaluate_expr); run_axis, side_axis, count, run_sign, side_sign = stairs_layout(comp); steps = []
    for k in range(1, count):
        center = {run_axis: val[run_axis] + run_sign * (k - 0.5) * val['g'], side_axis: val[side_axis] + side_sign * val['l'] / 2, 'y': val['y0'] + k * val['r'] / 2}
        size = {run_axis: val['g'], side_axis: val['l'], 'y': k * val['r']}
        steps.append(box_shape([center[a] for a in 'xyz'], [size[a] for a in 'xyz'], part=k))
    return steps


BOX_NATURES = {'meuble': ((0.8, 0.4, 0.8), '#c19a6b'), 'placard': ((1.0, 0.6, 2.4), '#9e9e9e'), 'poteau': ((0.1, 0.1, 2.5), '#707070'),
               'potager': ((1.2, 2.4, 0.3), '#6b8e23'), 'lit': ((1.4, 1.9, 0.5), '#b4a7d6')}


def box_nature(comp): return BOX_NATURES.get(comp['options'].get('nature'), BOX_NATURES['meuble'])


def build_box(comp, quantity, initial, components):
    comp['options']['nature'] = str(comp['options'].get('nature') or 'meuble'); (size_x, size_z, height), _ = box_nature(comp)
    quantity('x0', 'x', 'coin d’origine (x)', 0.0); quantity('z0', 'z', 'coin d’origine (z)', 0.0)
    quantity('lx', 'dim', 'dimension x', size_x); quantity('lz', 'dim', 'dimension z', size_z); quantity('h', 'dim', 'hauteur', height); quantity('y0', 'y', 'base', '0')
    quantity('rot', 'angle', 'rotation autour du coin d’origine', 0.0)


def rotation(comp):
    expr = comp['expressions']; angle = expr.get('rot', Expr()); k = angle.constant()
    cos, sin = (Expr.of(round(f(math.radians(k)), 12)) for f in (math.cos, math.sin)) if k is not None else (call(f, angle * (math.pi / 180)) for f in ('cos', 'sin'))
    return lambda u, v: {'x': expr['x0'] + u * cos + v * sin, 'z': expr['z0'] - u * sin + v * cos}


def box_corner(comp):
    expr = comp['expressions']; at = rotation(comp)
    return lambda i, j: at(i * expr['lx'], j * expr['lz'])


def box_anchors(comp):
    expr = comp['expressions']; corner = box_corner(comp)
    faces = {f'{axis}{k}': corner(k, 0.5) if axis == 'x' else dict(reversed(corner(0.5, k).items())) for axis in 'xz' for k in (0, 1)}
    return {**faces, 'bas': {'y': expr['y0']}, 'haut': {'y': expr['y0'] + expr['h']},
            'centre': {**corner(0.5, 0.5), 'y': expr['y0'] + 0.5 * expr['h']}, **{f'coin_{i}{j}': corner(i, j) for i in (0, 1) for j in (0, 1)}}


def box_geometry(comp, evaluate_expr, walls):
    val = evaluated(comp, evaluate_expr); center = box_anchors(comp)['centre']
    return [box_shape([evaluate_expr(center[a]) for a in 'xyz'], [val['lx'], val['h'], val['lz']], math.radians(val.get('rot', 0.0)), color=box_nature(comp)[1])]


def build_roof(comp, quantity, initial, components):
    quantity('x0', 'x', 'coin d’origine (x)', 0.0); quantity('z0', 'z', 'coin d’origine (z)', 0.0)
    quantity('long', 'dim', 'longueur (le long du faîtage)', 10.0); width = quantity('larg', 'dim', 'largeur', 8.0)
    quantity('rot', 'angle', 'orientation autour du coin d’origine', 0.0)
    low, high = quantity('hg', 'y', 'hauteur de gouttière', 3.0), quantity('hf', 'y', 'hauteur de faîtage', 6.0)
    run = 0.5 * width + quantity('deb', 'dim', 'débord', 0.4)
    comp['expressions'].update(pente=call('atan2', high - low, run) * (180 / math.pi), egout=high - (high - low) * (0.5 * width) / run)


def roof_anchors(comp):
    expr = comp['expressions']; at = rotation(comp); over, half, end = expr['deb'], 0.5 * expr['larg'], expr['long']
    def lift(u, v, y): return {**at(u, v), 'y': expr[y]}
    return {'faitage_a': lift(-over, half, 'hf'), 'faitage_b': lift(end + over, half, 'hf'), 'gouttiere_1': lift(0.5 * end, -over, 'hg'),
            'gouttiere_2': lift(0.5 * end, expr['larg'] + over, 'hg'), 'pignon_a': lift(0, half, 'hf'), 'pignon_b': lift(end, half, 'hf'),
            **{f'coin_{i}{j}': at(i * end, j * expr['larg']) for i in (0, 1) for j in (0, 1)}}


def roof_geometry(comp, evaluate_expr, walls):
    expr = comp['expressions']; at = rotation(comp); over, half, end = expr['deb'], 0.5 * expr['larg'], expr['long']
    def point(u, v, y): xz = at(u, v); return [evaluate_expr(xz['x']), evaluate_expr(expr[y]), evaluate_expr(xz['z'])]
    ridge = [point(end + over, half, 'hf'), point(-over, half, 'hf')]
    pans = [[point(-over, v, 'hg'), point(end + over, v, 'hg'), *ridge] for v in (-over, expr['larg'] + over)]
    gables = [[point(u, 0, 'egout'), point(u, expr['larg'], 'egout'), point(u, half, 'hf')] for u in (0, end)]
    return ([{'kind': 'mesh', 'vertices': pan, 'faces': [[0, 1, 2], [0, 2, 3]], 'part': f'pan_{k}'} for k, pan in enumerate(pans, 1)] +
            [{'kind': 'mesh', 'vertices': gable, 'faces': [[0, 1, 2]], 'part': f'pignon_{k}', 'color': TYPES['mur'].color} for k, gable in zip('ab', gables)])


ComponentType = namedtuple('ComponentType', 'label build anchors geometry color options')
OPENING_OPTIONS = {'mur': 'mur porteur'}
LEVEL_OPTIONS = {'contour': 'contour [[x, z], …]'}
TYPES = {
    'mur': ComponentType('Mur', build_wall, wall_anchors, wall_geometry, '#d9d4c7', {'axe': "parallèle à 'x' ou 'z' ; vide : en biais", 'sens': 'côté de l’épaisseur (1 ou -1)'}),
    'porte': ComponentType('Porte', build_opening, opening_anchors, opening_geometry, '#8b5a2b', OPENING_OPTIONS),
    'fenetre': ComponentType('Fenêtre', build_opening, opening_anchors, opening_geometry, '#7fb3d5', OPENING_OPTIONS),
    'porte_fenetre': ComponentType('Porte-fenêtre', build_opening, opening_anchors, opening_geometry, '#5d9ccf', OPENING_OPTIONS),
    'sol': ComponentType('Sol', build_level, polygon_anchors, polygon_geometry, '#b0a18a', LEVEL_OPTIONS),
    'plafond': ComponentType('Plafond', build_level, polygon_anchors, polygon_geometry, '#f2f2f2', LEVEL_OPTIONS),
    'escalier': ComponentType('Escalier droit', build_stairs, stairs_anchors, stairs_geometry, '#a0522d',
                              {'axe': "montée selon 'x' ou 'z'", 'sens': 'sens de montée (1 ou -1)', 'cote': 'côté de la largeur (1 ou -1)', 'n': 'nombre de hauteurs'}),
    'boite': ComponentType('Boîte', build_box, box_anchors, box_geometry, None, {'nature': f'texte libre ({", ".join(BOX_NATURES)}, …)'}),
    'parcelle': ComponentType('Zone extérieure', build_parcel, polygon_anchors, polygon_geometry, '#6aa84f', {'sommets': 'sommets [[x, z], …]'}),
    'toit': ComponentType('Toit', build_roof, roof_anchors, roof_geometry, '#b5523b', {}),
}
PRIORS = {'mur': {'ep': (0.2, 0.1), 'h': (2.5, 0.2)}, 'porte': {'h': (2.04, 0.05)}, 'fenetre': {'al': (0.9, 0.2)}, 'porte_fenetre': {'h': (2.15, 0.1)},
          'plafond': {'y': (2.5, 0.2)}, 'escalier': {'g': (0.25, 0.04), 'r': (0.175, 0.02)}, 'boite': {'rot': (None, 2.0)},
          'toit': {'pente': (35.0, 15.0), 'deb': (0.4, 0.3), 'rot': (None, 2.0)}}
ANGLE_PRIORS = ('rot', 'pente')
ANCHOR_LABELS = {'a': 'extrémité a', 'b': 'extrémité b', 'a_ext': 'extrémité a, face opposée', 'b_ext': 'extrémité b, face opposée', 'face': 'face',
                 'face_ext': 'face opposée', 'g': 'tableau côté a', 'd': 'tableau côté b', 'g_ext': 'tableau côté a, face opposée',
                 'd_ext': 'tableau côté b, face opposée', 'depart': '1re contremarche', 'arrivee': 'dernière contremarche', 'cote': 'côté',
                 'cote_ext': 'côté opposé', 'x0': 'milieu de la face x min', 'x1': 'milieu de la face x max', 'z0': 'milieu de la face z min',
                 'z1': 'milieu de la face z max', 'coin_00': 'coin d’origine', 'faitage_a': 'faîtage, extrémité a', 'faitage_b': 'faîtage, extrémité b',
                 'gouttiere_1': 'gouttière côté origine', 'gouttiere_2': 'gouttière opposée', 'pignon_a': 'sommet du pignon a', 'pignon_b': 'sommet du pignon b'}


def fields(type):
    found = []
    def quantity(name, axis, label, default):
        found.append({'name': name, 'label': label, 'unit': '°' if axis == 'angle' else 'm', 'default': default})
        return Expr({name: 1.0})
    wall = {'type': 'mur', 'expressions': dict.fromkeys(WALL_FRAME, Expr())}
    TYPES[type].build({'id': '', 'type': type, 'options': {'mur': 'mur'}, 'expressions': {}}, quantity, lambda expr: 1.0, {'mur': wall})
    return found


def types():
    return [{'type': name, 'label': kind.label, 'color': kind.color, 'options': {**kind.options, 'a_priori': f'observations « {PRIOR_STATUS} »'},
             'fields': fields(name)} for name, kind in TYPES.items()]


def load(project):
    project.db.execute(SCHEMA); components = {}
    for comp_id, data in project.db.execute('select id, json from components order by rowid'):
        comp = json.loads(data); comp['expressions'] = {k: Expr(v) for k, v in comp['expressions'].items()}; components[comp_id] = comp
    return components


def add(project, type, id, options=None, label=None):
    if type not in TYPES: raise ValueError(f'type inconnu : {type} ({", ".join(TYPES)})')
    components = load(project)
    if not COMPONENT_ID.fullmatch(str(id)) or id in components: raise ValueError(f'identifiant invalide ou déjà pris : {id!r}')
    options = dict(options or {}); kind = TYPES[type]; existing = project.params(); created, labels = [], {}
    default_label = str(options.get('nature') or 'meuble').capitalize() if type == 'boite' else kind.label
    comp = {'id': id, 'type': type, 'label': label or f'{default_label} {id}', 'options': options, 'expressions': {}}

    def quantity(name, axis, quantity_label, default):
        labels[name] = quantity_label; value = options.get(name)
        value = default if value is None else value
        if isinstance(value, str): expr = Expr.of(value)
        else:
            param = f'{id}_{name}'
            if param in existing: raise ValueError(f'paramètre déjà existant : {param}')
            created.append((param, float(value), axis, f"{comp['label']} — {quantity_label}")); expr = Expr({param: 1.0})
        comp['expressions'][name] = expr
        return expr

    def initial(expr): return evaluate(expr, {**{n: v[0] for n, v in existing.items()}, **{n: v for n, v, *_ in created}})

    kind.build(comp, quantity, initial, components); own = {n for n, *_ in created}
    priors = [(name, comp['expressions'][name], value, sigma) for name, (value, sigma) in PRIORS.get(type, {}).items()
              if options.get('a_priori', True) and own & set(NAME.findall(str(comp['expressions'][name])))]
    index = {n: i for i, n in enumerate([*existing, *(n for n, *_ in created)])}
    for expr in comp['expressions'].values(): compile_expr(str(expr), index)
    comp['params'] = [n for n, *_ in created]; source = f'comp:{id}'; db = project.db
    db.executemany('insert into params values (?,?,?,?)', created)
    db.executemany('insert or replace into obs values (?,?,?,?,?,?,?)',
                   [(f'{source}:{id}_{name}', source, str(expr), initial(expr) if value is None else value, sigma, PRIOR_STATUS,
                     f"{comp['label']} — {labels.get(name, name)} (a priori)") for name, expr, value, sigma in priors])
    db.execute('insert into components values (?,?,?)', (id, type, json.dumps(comp, ensure_ascii=False))); db.commit()
    return comp


def remove(project, comp_id):
    components = load(project); comp = components.pop(comp_id, None)
    if comp is None: raise ValueError(f'composant inconnu : {comp_id}')
    own, source = set(comp['params']), f'comp:{comp_id}'
    users = [k for k, other in components.items() if own & param_names(other)]
    users += [o.id for o in project.observations() if o.source != source and own & set(NAME.findall(o.expr))]
    if users: raise ValueError(f'{comp_id} est utilisé par : {", ".join(users[:5])}' + (f', … ({len(users)} au total)' if len(users) > 5 else ''))
    db = project.db; db.executemany('delete from params where name=?', [(n,) for n in own])
    db.execute('delete from obs where source=?', (source,)); db.execute('delete from components where id=?', (comp_id,)); db.commit()


def obs_unit(obs): return '°' if obs.source.startswith('comp:') and obs.id.rpartition('_')[2] in ANGLE_PRIORS else 'm'


def summary(comp): return {**{k: comp[k] for k in ('id', 'type', 'label', 'options', 'params')}, 'anchors': list(anchors(comp))}
def anchors(comp): return TYPES[comp['type']].anchors(comp)


def anchor(comp, name):
    found = anchors(comp)
    if name not in found: raise ValueError(f"{comp['id']} : ancre inconnue {name!r} ({', '.join(found)})")
    return found[name]


def measure_expr(start, end, axis='dist', values=None):
    start, end = (anchor(*ref) if isinstance(ref, tuple) else ref for ref in (start, end))
    axes = [a for a in 'xyz' if a in start and a in end] if axis == 'dist' else [axis]
    if not axes or any(a not in start or a not in end for a in axes): raise ValueError(f'ancres sans coordonnée commune ({axis})')
    deltas = [d for d in (Expr.of(end[a]) - start[a] for a in axes) if d]
    if not any(set(d) - {'1'} for d in deltas): raise ValueError('cote sans paramètre : les deux ancres sont fixes')
    if len(deltas) == 3: expr = call('hypot', call('hypot', deltas[0], deltas[1]), deltas[2])
    elif len(deltas) == 2: expr = call('hypot', *deltas)
    elif values is not None: expr = -deltas[0] if evaluate(deltas[0], values) < 0 else deltas[0]
    else: expr = call('abs', deltas[0]) if axis == 'dist' else deltas[0]
    return str(expr)


def param_values(project, result=None):
    values = {n: (init, math.inf) for n, (init, _, _) in project.params().items()}
    result = project.model() if result is None else result
    if isinstance(result, dict): rows = [(n, p['value'], p['sigma']) for n, p in result['params'].items()]
    else: rows = zip(result.names, result.x, result.sigma) if result is not None else []
    for name, value, sigma in rows:
        if name in values: values[name] = (float(value), math.inf if sigma is None else float(sigma))
    return values


def geometry(project, result=None):
    values = param_values(project, result); index = {n: i for i, n in enumerate(values)}
    vector = np.array([v for v, _ in values.values()], float); cache = {}
    to_metres = {n: math.pi / 180 for n, (_, axis, _) in project.params().items() if axis == 'angle'}

    def evaluate_expr(expr):
        text = str(Expr.of(expr))
        if text not in cache: cache[text] = compile_expr(text, index)[0](vector)
        return cache[text]

    components = load(project); walls = [c for c in components.values() if c['type'] == 'mur']; shapes = []
    for comp in components.values():
        kind = TYPES[comp['type']]; sigma = max((values[n][1] * to_metres.get(n, 1.0) for n in param_names(comp) if n in values), default=0.0)
        common = {'id': comp['id'], 'type': comp['type'], 'label': comp['label'], 'color': kind.color, 'sigma_max': sigma if math.isfinite(sigma) else None}
        shapes += [{**common, **shape} for shape in kind.geometry(comp, evaluate_expr, walls)]
    return shapes


def clip(polygon, convex):
    for a, b in zip(convex, convex[1:] + convex[:1]):
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        points, polygon = polygon, []
        for p, q in zip(points, points[1:] + points[:1]):
            sp, sq = side(p), side(q)
            if sp >= 0: polygon.append(p)
            if sp * sq < 0: polygon.append((p[0] + sp / (sp - sq) * (q[0] - p[0]), p[1] + sp / (sp - sq) * (q[1] - p[1])))
    return polygon


def overlap_area(first, second):
    first, second, total = [tuple(p) for p in first], [tuple(p) for p in second], 0.0
    for a, b in zip(first[1:], first[2:]):
        triangle = [first[0], a, b]; orientation = cadastre.signed_area(triangle)
        if orientation: total += math.copysign(cadastre.area(clip(second, triangle if orientation > 0 else triangle[::-1])), orientation)
    return abs(total)


def footprint(shape):
    if shape['kind'] == 'polygon': return shape['vertices']
    (x, _, z), (lx, _, lz), angle = shape['center'], shape['size'], shape['rotY']; c, s = math.cos(angle) / 2, math.sin(angle) / 2
    return [(x + i * lx * c + j * lz * s, z - i * lx * s + j * lz * c) for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def inclusion_warnings(project, result):
    land = cadastre.geometry(project.folder, result)['entites']; land = [e for e in land if e['georeference']] or land
    parcel = next((e for e in land if e['type'] == 'parcelle' and e['principal']), None)
    if parcel is None: return []
    buildings, warnings = [e for e in land if e['type'] == 'bati' and e['observe']], []
    def warn(code, shape, area, where): warnings.append({'code': code, 'component': shape['id'], 'aire': round(area, 4), 'message': f"{shape['label']} déborde de {format(area, '.2f').replace('.', ',')} m² {where}"})
    for shape in (s for s in geometry(project, result) if s['type'] in ('boite', 'parcelle')):
        points = footprint(shape); total = cadastre.area(points)
        if (outside := total - overlap_area(points, parcel['pts'])) > AREA_TOLERANCE: warn('hors_parcelle', shape, outside, f"hors de la parcelle principale {parcel['label'] or parcel['id']}")
        on, building = max(((overlap_area(points, b['pts']), b) for b in buildings), key=lambda t: t[0], default=(0.0, None))
        if AREA_TOLERANCE < on < total - AREA_TOLERANCE: warn('sur_batiment', shape, on, f"sur le bâtiment {building['nature']} {building['label'] or building['id']}")
    return warnings


def resolve_ref(components, ref):
    comp_id, _, name = str(ref).partition(':')
    if comp_id not in components: raise ValueError(f'composant inconnu : {comp_id}')
    return components[comp_id], name


@rpc('components.types')
def rpc_types(ctx): return types()


@rpc('components.add')
def rpc_add(ctx, type, id, options=None, label=None): return summary(add(ctx.require_project(), type, id, options, label))


@rpc('components.list')
def rpc_list(ctx): return [summary(comp) for comp in load(ctx.require_project()).values()]


@rpc('components.remove')
def rpc_remove(ctx, id):
    remove(ctx.require_project(), id)
    return {'removed': id}


@rpc('components.anchors')
def rpc_anchors(ctx, id):
    comp, _ = resolve_ref(load(ctx.require_project()), id)
    label = lambda n: ANCHOR_LABELS.get(n) or re.sub(r'^s(\d+)$', r'sommet \1', n).replace('_', ' ')
    return [{'name': n, 'ref': f'{id}:{n}', 'label': label(n), 'axes': {a: str(e) for a, e in point.items()}} for n, point in anchors(comp).items()]


@rpc('components.measure_expr')
def rpc_measure(ctx, a, b, axis='dist'):
    project = ctx.require_project(); components = load(project)
    values = {n: v for n, (v, _) in param_values(project, ctx.result).items()}
    expr = measure_expr(resolve_ref(components, a), resolve_ref(components, b), axis, values)
    return {'expr': expr, 'value': evaluate(expr, values), 'label': f'{a} → {b}' + ('' if axis == 'dist' else f' ({axis})')}


@rpc('components.geometry')
def rpc_geometry(ctx): return geometry(ctx.require_project(), ctx.result)

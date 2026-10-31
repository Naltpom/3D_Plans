"""Solveur : moindres carrés robustes pondérés par 1/σ², marges, contradictions, prochaine meilleure mesure."""
import ast
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.optimize import least_squares

FUNCS = {'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'atan2': np.arctan2, 'hypot': np.hypot, 'abs': np.abs}
NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant, ast.Call,
         ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
STATUS = ('mesuré', 'lu', 'photo', 'déduit', 'supposé', 'ia')
W_MAX, R_MIN = 3.0, 0.1   # seuil de contradiction (résidu normalisé) ; redondance minimale pour pouvoir contrôler


@dataclass
class Obs:
    id: str
    expr: str      # fonction des paramètres, ex. "w2_x - end_x" ou "w4_z / s"
    value: float
    sigma: float
    source: str = ''
    status: str = 'mesuré'
    label: str = ''


def compile_expr(expr, index):
    """Expression restreinte (opérations, fonctions de FUNCS, paramètres) → (f(x), indices des paramètres utilisés)."""
    tree = ast.parse(expr, mode='eval'); used = set()
    for n in ast.walk(tree):
        if not isinstance(n, NODES): raise ValueError(f'{expr!r} : {type(n).__name__} interdit')
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCS): raise ValueError(f'{expr!r} : appel interdit')
        if isinstance(n, ast.Name) and n.id not in FUNCS:
            if n.id not in index: raise KeyError(f'{expr!r} : paramètre inconnu {n.id}')
            used.add(n.id)
    # eval sûr : arbre limité aux nœuds de NODES, appels limités à FUNCS, aucun builtin
    code = compile(tree, expr, 'eval'); env = {'__builtins__': {}, **FUNCS}
    return (lambda x: float(eval(code, env, {k: x[index[k]] for k in used}))), sorted(index[k] for k in used)


@dataclass
class Result:
    names: list
    x: np.ndarray
    sigma: np.ndarray          # inf = non déterminé
    C: np.ndarray
    obs: list
    resid: np.ndarray          # f(x) − valeur, unités de l'observation
    w: np.ndarray              # résidu normalisé (Baarda)
    redundancy: np.ndarray     # part contrôlée de chaque observation (0 = non contrôlable)
    chi2: float
    dof: int
    _idx: dict = field(repr=False, default_factory=dict)
    timing: dict = field(default_factory=dict)
    deps: list = field(repr=False, default_factory=list)

    def value(self, name): return float(self.x[self._idx[name]]), float(self.sigma[self._idx[name]])

    @property
    def contradictions(self):
        return [(o, float(w)) for o, w, r in zip(self.obs, self.w, self.redundancy) if abs(w) > W_MAX and r > R_MIN]

    def conflicts(self):
        """Contradictions regroupées : deux observations signalées sont liées si elles partagent un paramètre non global."""
        fl = [(i, w) for i, (w, r) in enumerate(zip(self.w, self.redundancy)) if abs(w) > W_MAX and r > R_MIN]
        cnt = np.bincount([j for d in self.deps for j in d], minlength=len(self.x))
        loc = [set(j for j in d if cnt[j] <= 0.3 * len(self.obs)) for d in self.deps]
        groups = []
        for i, w in fl:
            hit = [g for g in groups if any(loc[i] & loc[k] for k, _ in g)]
            for g in hit: groups.remove(g)
            groups.append(sum(hit, []) + [(i, w)])
        return [sorted([(self.obs[i], float(w)) for i, w in g], key=lambda t: -abs(t[1])) for g in groups]

    def grad(self, expr):
        f, dep = compile_expr(expr, self._idx); g = np.zeros(len(self.x))
        for j in dep:
            h = 1e-6 * max(1.0, abs(self.x[j])); e = np.zeros_like(self.x); e[j] = h
            g[j] = (f(self.x + e) - f(self.x - e)) / (2 * h)
        return f(self.x), g

    def predict(self, expr):
        v, g = self.grad(expr); return v, float(np.sqrt(max(g @ self.C @ g, 0.0)))


def jacobian(fun, x, deps, m):
    J = np.zeros((m, len(x)))
    for j in {j for d in deps for j in d}:
        h = 1e-6 * max(1.0, abs(x[j])); e = np.zeros_like(x); e[j] = h
        J[:, j] = (fun(x + e) - fun(x - e)) / (2 * h)
    return J


def solve(params, obs, loss='soft_l1'):
    """params : {nom: valeur initiale}. Retourne Result (marges issues de la covariance, résidus normalisés)."""
    names = list(params); idx = {n: i for i, n in enumerate(names)}
    fs = [compile_expr(o.expr, idx) for o in obs]
    y = np.array([o.value for o in obs], float); sg = np.array([o.sigma for o in obs], float)
    res = lambda x: (np.array([f(x) for f, _ in fs]) - y) / sg
    sol = least_squares(res, np.array([params[n] for n in names], float), loss=loss, f_scale=1.0, x_scale='jac')
    x = sol.x; r = res(x); J = jacobian(res, x, [d for _, d in fs], len(obs))
    wr = (1 + r ** 2) ** -0.25 if loss == 'soft_l1' else np.ones_like(r)        # √poids robustes (IRLS)
    Jw = J * wr[:, None]
    C = np.linalg.pinv(Jw.T @ Jw, rcond=1e-10)
    free = np.diag(np.eye(len(x)) - C @ (Jw.T @ Jw)) > 1e-6                       # directions non contraintes
    sigma = np.where(free, np.inf, np.sqrt(np.clip(np.diag(C), 0, None)))
    C0 = np.linalg.pinv(J.T @ J, rcond=1e-10)
    red = 1 - np.einsum('ij,jk,ik->i', J, C0, J)                                 # redondance (moindres carrés simples)
    h = np.einsum('ij,jk,ik->i', Jw, C, Jw)
    w = r / np.sqrt(np.clip(1 - h, 1e-9, None))
    dof = len(obs) - int(np.linalg.matrix_rank(J))
    return Result(names, x, sigma, C, obs, r * sg, w, np.clip(red, 0, 1), float(r @ r), dof, idx, deps=[d for _, d in fs])


def variance_factors(res, key):
    """Par groupe d'observations : σ réel / σ déclaré estimé sur les résidus (None si redondance < 1)."""
    g = {}
    for o, r, q in zip(res.obs, res.resid, res.redundancy):
        a = g.setdefault(key(o), [0.0, 0.0, 0]); a[0] += (r / o.sigma) ** 2; a[1] += q; a[2] += 1
    return {k: (float(np.sqrt(a[0] / a[1])) if a[1] >= 1 else None, a[1], a[2]) for k, a in g.items()}


def calibrate(res, key, lo=0.5, hi=2.0):
    """Observations avec σ multiplié par le facteur estimé de leur groupe (borné à [lo, hi])."""
    f = {k: v[0] for k, v in variance_factors(res, key).items()}
    return [replace(o, sigma=o.sigma * float(np.clip(f[key(o)] or 1.0, lo, hi))) for o in res.obs]


def next_best(res, candidates, targets=None, k=5):
    """candidates : [(libellé, expr, σ)]. Classe par baisse totale des marges (m) sur targets (défaut : tous)."""
    t = [res._idx[n] for n in (targets or res.names) if np.isfinite(res.sigma[res._idx[n]])]
    out = []
    for lab, expr, s in candidates:
        _, g = res.grad(expr); Cg = res.C @ g; v = g @ Cg
        if v <= 0: continue
        new = np.sqrt(np.clip(res.sigma[t] ** 2 - Cg[t] ** 2 / (v + s ** 2), 0, None))
        out.append((float((res.sigma[t] - new).sum()), lab, expr, float(np.sqrt(v))))
    return sorted(out, reverse=True)[:k]

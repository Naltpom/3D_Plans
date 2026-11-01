import ast, re
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

FUNCS = {'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'atan2': np.arctan2, 'hypot': np.hypot, 'abs': np.abs}
NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant, ast.Call,
         ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
STATUS = ('mesuré', 'lu', 'photo', 'déduit', 'supposé', 'ia')
ENV = {'__builtins__': {}, **FUNCS}
NAME = re.compile(r'(?<![\w.])[^\W\d]\w*')
MAX_NFEV = 200
W_MAX, R_MIN, W_NEAR = 3.0, 0.1, 2.0


@dataclass
class Obs:
    id: str
    expr: str
    value: float
    sigma: float
    source: str = ''
    status: str = 'mesuré'
    label: str = ''


def parse(expr, index):
    tree = ast.parse(expr, mode='eval'); used = []
    for n in ast.walk(tree):
        if not isinstance(n, NODES): raise ValueError(f'{expr!r} : {type(n).__name__} interdit')
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCS): raise ValueError(f'{expr!r} : appel interdit')
        if isinstance(n, ast.Name) and n.id not in FUNCS:
            if n.id not in index: raise KeyError(f'{expr!r} : paramètre inconnu {n.id}')
            if n.id not in used: used.append(n.id)
    return tree, used


def compile_expr(expr, index):
    tree, used = parse(expr, index)
    code = compile(tree, expr, 'eval')
    return (lambda x: float(eval(code, ENV, {k: x[index[k]] for k in used}))), sorted(index[k] for k in used)


def batch(exprs, index):
    groups, deps = {}, []
    for i, e in enumerate(exprs):
        used = [n for n in dict.fromkeys(NAME.findall(e)) if n not in FUNCS]
        key = NAME.sub(lambda m: m[0] if m[0] in FUNCS else f'_{used.index(m[0])}', e)
        if key not in groups:
            if set(parse(e, index)[1]) != set(used): raise ValueError(f'{e!r} : expression non reconnue')
            groups[key] = (compile(parse(key, {f'_{k}': k for k in range(len(used))})[0], '<obs>', 'eval'), [], [])
        try: cols = [index[n] for n in used]
        except KeyError as err: raise KeyError(f'{e!r} : paramètre inconnu {err.args[0]}') from None
        groups[key][1].append(i); groups[key][2].append(cols); deps.append(sorted(cols))
    return [(code, np.array(r), np.array(c, int).reshape(len(r), -1)) for code, r, c in groups.values()], deps


def evaluate(groups, x, m):
    out = np.empty(m)
    for code, rows, cols in groups: out[rows] = eval(code, ENV, {f'_{k}': x[j] for k, j in enumerate(cols.T)})
    return out


@dataclass
class Result:
    names: list
    x: np.ndarray
    sigma: np.ndarray
    C: np.ndarray
    obs: list
    resid: np.ndarray
    w: np.ndarray
    redundancy: np.ndarray
    chi2: float
    dof: int
    J: csr_array = field(repr=False, default=None)
    _idx: dict = field(repr=False, default_factory=dict)
    timing: dict = field(default_factory=dict)
    deps: list = field(repr=False, default_factory=list)
    converged: bool = True

    def value(self, name): return float(self.x[self._idx[name]]), float(self.sigma[self._idx[name]])

    @property
    def contradictions(self):
        return [(o, float(w)) for o, w, r in zip(self.obs, self.w, self.redundancy) if abs(w) > W_MAX and r > R_MIN]

    def conflicts(self):
        m, n, aw, ok = len(self.obs), len(self.x), np.abs(self.w), self.redundancy > R_MIN
        cnt = np.bincount([j for d in self.deps for j in d], minlength=n)
        nsrc = np.bincount([j for j, _ in {(j, o.source) for o, d in zip(self.obs, self.deps) for j in d}], minlength=n)
        loc = [{j for j in d if not (cnt[j] > 5 and (nsrc[j] >= 2 or cnt[j] > 0.1 * m))} for d in self.deps]
        groups = []
        for i in np.flatnonzero((aw > W_MAX) & ok):
            hit = [g for g in groups if any(loc[i] & loc[k] for k in g)]
            for g in hit: groups.remove(g)
            groups.append(sum(hit, []) + [i])
        near = np.flatnonzero((aw >= W_NEAR) & (aw <= W_MAX) & ok)
        groups = [g + [p for p in near if any(loc[p] & loc[k] for k in g)] for g in groups]
        return [sorted([(self.obs[i], float(self.w[i])) for i in g], key=lambda t: -abs(t[1])) for g in groups]

    def grad(self, expr):
        groups = batch([expr], self._idx)[0]
        return float(evaluate(groups, self.x, 1)[0]), jacobian(groups, self.x, np.ones(1)).toarray()[0]

    def predict(self, expr):
        v, g = self.grad(expr); return v, float(np.sqrt(max(g @ self.C @ g, 0.0)))


def jacobian(groups, x, sg):
    h = 1e-6 * np.maximum(1.0, np.abs(x)); r, c, v = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0)]
    for code, rows, cols in groups:
        loc = {f'_{k}': x[j] for k, j in enumerate(cols.T)}
        for k, j in enumerate(cols.T):
            a, b = (eval(code, ENV, {**loc, f'_{k}': x[j] + s * h[j]}) for s in (1, -1))
            r.append(rows); c.append(j); v.append((a - b) / (2 * h[j] * sg[rows]))
    return csr_array((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(len(sg), len(x)))


def eig_pinv(B, rcond):
    l, U = np.linalg.eigh(B); keep = l > rcond * l[..., -1:]
    return (U * np.divide(1, l, out=np.zeros_like(l), where=keep)[..., None, :]) @ np.swapaxes(U, -1, -2), keep.all(-1)


def chol_inv(B, rcond):
    try: Ci = cho_solve(cho_factor(B), np.eye(len(B)))
    except np.linalg.LinAlgError: return None, False
    return (Ci if np.abs(B).sum(1).max() * np.abs(Ci).sum(1).max() * rcond <= 1 else None), True


def scale(A):
    d = A.diagonal(); return np.sqrt(d) + (d == 0)


def pinv(A, rcond=1e-10):
    d = scale(A); D = A.toarray() / np.outer(d, d); C = np.zeros_like(D); weak = []; lab = connected_components(A, directed=False)[1]
    blocks = np.split(np.argsort(lab, kind='stable'), np.cumsum(np.bincount(lab))[:-1])
    small = [np.array([b for b in blocks if len(b) == s]) for s in {len(b) for b in blocks} if s <= 64]
    for P in small + [b[None] for b in blocks if len(b) > 64]:
        ix = P[:, :, None], P[:, None, :]; Ci, chol = chol_inv(D[ix][0], rcond) if P.shape[1] > 64 else (None, False)
        if Ci is None: Ci, full = eig_pinv(D[ix], rcond); weak += list(P[~(full | chol)])
        C[ix] = Ci
    return C / np.outer(d, d), weak


def solve(params, obs, loss='soft_l1'):
    names = list(params); idx = {n: i for i, n in enumerate(names)}
    groups, deps = batch([o.expr for o in obs], idx)
    y = np.array([o.value for o in obs], float); sg = np.array([o.sigma for o in obs], float)
    res = lambda x: (evaluate(groups, x, len(obs)) - y) / sg
    jac = lambda x: jacobian(groups, x, sg) if len(names) > 100 else jacobian(groups, x, sg).toarray()
    sol = least_squares(res, np.array([params[n] for n in names], float), jac=jac, loss=loss, f_scale=1.0, x_scale='jac', tr_options={'atol': 1e-12, 'btol': 1e-12}, max_nfev=MAX_NFEV)
    x = sol.x; r = res(x); J = jacobian(groups, x, sg)
    wr = (1 + r ** 2) ** -0.25 if loss == 'soft_l1' else np.ones_like(r)
    Jw = J.multiply(wr[:, None]).tocsr(); A = Jw.T @ Jw
    A0 = J.T @ J; (C, _), (C0, weak) = pinv(A), pinv(A0)
    rank = len(x) - sum(len(b) - int(np.linalg.matrix_rank(J[:, b].toarray() / scale(A0)[b])) for b in weak)
    lev = lambda K, M: np.asarray(K.multiply(M).sum(1)).ravel()
    sigma = np.where(1 - lev(A, C) > 1e-6, np.inf, np.sqrt(np.clip(np.diag(C), 0, None)))
    w = r / np.sqrt(np.clip(1 - lev(Jw, Jw @ C), 1e-9, None))
    return Result(names, x, sigma, C, obs, r * sg, w, np.clip(1 - lev(J, J @ C0), 0, 1), float(r @ r), len(obs) - rank, J, idx, deps=deps, converged=bool(sol.status > 0))


def variance_factors(res, key):
    g = {}
    for o, r, q in zip(res.obs, res.resid, res.redundancy):
        a = g.setdefault(key(o), [0.0, 0.0, 0]); a[0] += (r / o.sigma) ** 2; a[1] += q; a[2] += 1
    return {k: (float(np.sqrt(a[0] / a[1])) if a[1] >= 1 else None, a[1], a[2]) for k, a in g.items()}


def next_best(res, candidates, targets=None, k=5):
    t = [res._idx[n] for n in (targets or res.names) if np.isfinite(res.sigma[res._idx[n]])]
    G = jacobian(batch([e for _, e, _ in candidates], res._idx)[0], res.x, np.ones(len(candidates)))
    s2, st = np.array([s for _, _, s in candidates], float) ** 2, res.sigma[t]; v, gain = np.zeros((2, len(candidates)))
    for i in range(0, len(candidates), 2048):
        g = G[i:i + 2048]; M = g @ res.C; v[i:i + 2048] = vi = np.asarray(g.multiply(M).sum(1)).ravel()
        M = np.square(M[:, t]); M /= np.where(vi > 0, vi + s2[i:i + 2048], np.inf)[:, None]
        gain[i:i + 2048] = st.sum() - np.sqrt(np.clip(np.subtract(st ** 2, M, out=M), 0, None, out=M), out=M).sum(1)
    ok = np.flatnonzero((v > 0) & (abs(G) @ np.isinf(res.sigma) == 0)); best = ok[np.lexsort((v[ok], np.round(gain[ok], 12)))[::-1][:k]]
    return [(float(gain[j]), *candidates[j][:2], float(np.sqrt(v[j]))) for j in best]

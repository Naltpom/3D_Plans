import json
import numpy as np
import pytest
from plans3d import Obs, solve, next_best, Project
from plans3d.core import compile_expr

TRUE = {'w1': 1.2, 'w2': 2.5, 'w3': 3.1, 'w4': 4.4, 'w5': 5.0, 's': 0.5}
INIT = {k: v + 0.1 for k, v in TRUE.items()}
ST, SU = 0.01, 0.02          # tape 1 cm ; photo 0.02 u (= 1 cm at s=0.5)
TAPE = ['w1', 'w2 - w1', 'w3 - w2', 'w4 - w3', 'w5 - w4', 'w3 - w1', 'w5 - w2']
PHOTO = [f'w{k} / s' for k in range(1, 6)]


def ev(e): return eval(e, {}, dict(TRUE))


def make_obs(rng, tape=TAPE, photo=PHOTO, su=None):
    su = su or {}
    o = [Obs(f't{i}', e, ev(e) + rng.normal(0, ST), ST) for i, e in enumerate(tape)]
    o += [Obs(f'p{i}', e, ev(e) + rng.normal(0, su.get(e, SU)), su.get(e, SU), status='photo') for i, e in enumerate(photo)]
    return o


IDX = {n: i for i, n in enumerate(['a_x', 'b_x', 's'])}


@pytest.mark.parametrize('e', ['a_x.real', 'sqrt.__class__', '__import__("os")', '(lambda: 1)()', 'zz + 1',
                               'open("f")', 'a_x if s else b_x', '[a_x][0]', 'len(a_x)'])
def test_compile_rejects(e):
    with pytest.raises((ValueError, KeyError)): compile_expr(e, IDX)


def test_compile_accepts():
    f, dep = compile_expr('hypot(a_x - b_x, 2) / s', IDX)
    assert dep == [0, 1, 2] and f(np.array([4., 1., 2.])) == pytest.approx(np.hypot(3, 2) / 2)


def test_chain_recovery_and_sigma():
    r = solve(INIT, make_obs(np.random.default_rng(0)))
    assert r.dof == 6
    for k, v in TRUE.items():
        x, s = r.value(k); assert np.isfinite(s) and abs(x - v) < 3 * s, (k, x, s)
    rng = np.random.default_rng(1); X, S = [], []
    for _ in range(200):
        q = solve(INIT, make_obs(rng)); X.append(q.x); S.append(q.sigma)
    ratio = np.mean(S, 0) / np.std(X, 0)
    assert np.all((ratio > 0.8) & (ratio < 1.25)), dict(zip(r.names, ratio.round(3)))


def test_outlier_and_no_redundancy():
    obs = make_obs(np.random.default_rng(2)); base = solve(INIT, obs)
    obs[2].value += 10 * ST                                   # 't2' : w3 - w2
    obs.append(Obs('solo', 'p7 - w5', 1.0, ST))               # seule obs de p7 : redondance 0
    r = solve({**INIT, 'p7': 0.0}, obs)
    assert [o.id for o, _ in r.contradictions] == ['t2'], [(o.id, w) for o, w in r.contradictions]
    assert r.redundancy[-1] < 1e-6 and r.value('p7')[0] == pytest.approx(r.value('w5')[0] + 1.0, abs=1e-6)
    for k in TRUE:   # soft_l1 borne l'influence à ~1σ : params de l'obs fautive < 1.5σ, autres < 0.5σ
        assert abs(r.value(k)[0] - base.value(k)[0]) < (1.5 if k in ('w2', 'w3') else 0.5) * base.value(k)[1], k


def test_undetermined():
    obs = make_obs(np.random.default_rng(3)) + [Obs('ab', 'a + b', 1.0, ST)]
    r = solve({**INIT, 'a': 0., 'b': 0., 'u': 0.}, obs)
    for k in 'abu': assert r.value(k)[1] == np.inf, k
    for k in TRUE: assert np.isfinite(r.value(k)[1]), k
    assert r.predict('a + b')[1] == pytest.approx(ST, rel=1e-3)
    assert next_best(r, [('u', 'u', ST), ('a', 'a', ST)]) is not None


def test_predict_and_next_best():
    r = solve(INIT, make_obs(np.random.default_rng(4))); i, C = r._idx, r.C
    v, s = r.predict('w3 - w1')
    assert v == pytest.approx(r.x[i['w3']] - r.x[i['w1']])
    assert s == pytest.approx(np.sqrt(C[i['w3'], i['w3']] + C[i['w1'], i['w1']] - 2 * C[i['w1'], i['w3']]), rel=1e-6)
    v, s = r.predict('w2 * s')
    g = np.zeros(len(r.x)); g[i['w2']] = r.x[i['s']]; g[i['s']] = r.x[i['w2']]
    assert s == pytest.approx(np.sqrt(g @ C @ g), rel=1e-5)
    # w5 : seulement une photo à 0.2 u (≈10 cm)
    r = solve(INIT, make_obs(np.random.default_rng(5), tape=TAPE[:4] + ['w3 - w1'], su={'w5 / s': 0.2}))
    assert max(r.names[:5], key=lambda n: r.value(n)[1]) == 'w5'
    cands = [('d21', 'w2 - w1', ST), ('d43', 'w4 - w3', ST), ('d54', 'w5 - w4', ST), ('d32', 'w3 - w2', ST)]
    assert next_best(r, cands)[0][1] == 'd54'


HDR = 'id;expr;value;sigma;unit;status;label\n'


def test_project_incremental(tmp_path):
    P = Project(str(tmp_path / 'proj'))
    for k in ('w1', 'w2', 'w3'): P.add_param(k, 1.0, 'x', k)
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text(HDR + 'd1;w1;120;1;cm;mesuré;\nd2;w2 - w1;130;1;cm;;\n', encoding='utf-8')
    b.write_text(HDR + 'd3;w3 - w2;60;1;cm;lu;x\nd4;w3;310;1;cm;mesuré;\n', encoding='utf-8')
    P.add_source('A', str(a), 'cotes_csv'); P.add_source('B', str(b), 'cotes_csv')
    r = P.update()
    assert sorted(r.timing['reextracted']) == ['A', 'B']
    m = json.loads((tmp_path / 'proj' / 'model.json').read_text(encoding='utf-8'))
    assert m['dof'] == 1 and len(m['obs']) == 4 and m['params']['w3']['value'] == pytest.approx(3.1, abs=1e-6)
    assert m['params']['w3']['sigma'] > 0 and sorted(m['params']['w1']['obs']) == ['A:d1', 'A:d2']
    assert {o['id']: o['status'] for o in m['obs']}['A:d2'] == 'mesuré'
    assert P.update().timing['reextracted'] == []
    b.write_text(HDR + 'd3;w3 - w2;60;1;cm;lu;x\nd4;w3;312;1;cm;mesuré;\n', encoding='utf-8')
    r = P.update()
    assert r.timing['reextracted'] == ['B'] and len(P.observations()) == 4
    assert {o.id: o.value for o in P.observations()}['B:d4'] == pytest.approx(3.12)
    assert r.value('w3')[0] > 3.1
    P.db.close()

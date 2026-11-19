import glob, os
from .rpc import project_meta, rpc

QUESTIONS = ('forme', 'longueurs', 'echelle', 'ouvertures', 'hauteur')
SOURCES = {
    'photos': ('Photos', {'forme', 'longueurs', 'ouvertures', 'hauteur'}),
    'scan': ('Scan au téléphone', {'forme', 'longueurs', 'ouvertures', 'hauteur'}),
    'croquis': ('Croquis à main levée', {'forme', 'ouvertures'}),
    'plan': ('Plan coté', {'forme', 'longueurs', 'echelle', 'ouvertures'}),
    'cadastre': ('Plan cadastral', {'forme', 'longueurs', 'echelle'}),
    'manuel': ('Construction manuelle', set(QUESTIONS)),
    'gabarit': ('Fiche constructeur', set(QUESTIONS)),
    'relief': ("Relevé d'altitudes", set()),
}
OFFERED = {'piece': ['photos', 'scan', 'croquis', 'plan', 'manuel', 'cadastre'], 'maison': ['cadastre', 'photos', 'scan', 'croquis', 'plan', 'manuel', 'relief'],
           'vehicule': ['gabarit', 'photos', 'scan', 'croquis', 'plan', 'manuel']}
PHOTOS, DRAWN, COPIED = ['photos', 'scan'], ['croquis', 'plan', 'manuel'], {'plan'}
FLOWS = {'piece': ('projet', 'sources', 'positions', 'murs', 'echelle', 'verifier', 'meubler'),
         'maison': ('projet', 'sources', 'cadastre', 'murs', 'relief', 'toit', 'facades', 'echelle', 'verifier', 'meubler'),
         'vehicule': ('projet', 'sources', 'gabarit', 'positions', 'murs', 'echelle', 'verifier', 'meubler')}
STEPS = {'projet': ('Le projet', 'Ce que tu reconstruis'), 'sources': ('Les sources', 'Ce que tu as sous la main'), 'positions': ('Positions des photos', 'Calcul automatique'),
         'cadastre': ('Le cadastre', 'Parcelle et murs extérieurs'), 'gabarit': ('Le gabarit', 'Fiche constructeur'), 'murs': ('Les murs', 'Valider ce qui est trouvé'),
         'echelle': ("L'échelle", 'Une mesure au mètre'), 'relief': ('Le relief', 'Optionnel : maison sur le terrain'), 'toit': ('Le toit', "Proposé sur l'emprise"),
         'facades': ('Les façades', 'Photos recalées'), 'verifier': ('Vérifier', 'Contradictions, mesure utile'), 'meubler': ('Meubler', 'Optionnel, une fois le plan prêt')}
HEIGHT = {'piece': 'La hauteur sous plafond', 'maison': 'La hauteur des murs', 'vehicule': 'La hauteur utile'}


def listed(names):
    return ', '.join(names[:-1]) + f' et {names[-1]}' if len(names) > 1 else ''.join(names)


def parcours(type, sources, walls=()):
    type = type if type in FLOWS else 'piece'
    chosen = [s for s in OFFERED[type] if s in sources]
    has = lambda *names: any(n in chosen for n in names)
    answers = {q: [s for s in chosen if q in SOURCES[s][1]] for q in QUESTIONS}
    missing = [q for q in QUESTIONS if not answers[q]]
    unsized = 'longueurs' in missing or 'echelle' in missing
    copied = not unsized and set(answers['echelle']) <= COPIED
    built = lambda s: s in DRAWN or s in PHOTOS and type != 'maison' or s == 'cadastre' and type == 'piece'
    useful = {'positions': has(*PHOTOS), 'facades': has(*PHOTOS), 'cadastre': has('cadastre'), 'gabarit': has('gabarit'), 'relief': has('relief'),
              'murs': any(built(s) for s in answers['forme']), 'echelle': bool(answers['forme']) and (unsized or copied)}
    labels = dict(STEPS)
    if has('croquis', 'plan'): labels['murs'] = ('Les murs', 'Lire le plan' + (' et le caler sur les photos' if has(*PHOTOS) else ''))
    elif has('manuel') and not has(*PHOTOS): labels['murs'] = ('Construire', 'Murs, portes, fenêtres, angles')
    elif type == 'vehicule' and has('gabarit'): labels['murs'] = ('Le recalage', 'Photos sur le gabarit')
    labels['echelle'] = ('Les longueurs', 'Chaque mur au mètre') if 'longueurs' in missing else ('Les cotes du plan', 'Recopier les cotes écrites') if copied else STEPS['echelle']
    if type == 'maison': labels.update(verifier=('Vérifier', 'Débordements, mesure utile'), meubler=('Meubler', 'Optionnel'),
                                       facades=('Les façades', 'Photos et scan recalés' if has('photos') else 'Scan au téléphone, recalé') if has('scan') else STEPS['facades'])
    if type == 'vehicule': labels.update(positions=('Les photos', 'Positions des photos'), verifier=('Cotes au mètre', 'Vérifier, mesure utile'), meubler=('Aménager', 'Galerie et agencement'))
    todo = ([f"La longueur de chaque mur : {listed(list(walls))}" if walls else 'La longueur de chaque mur, au mètre'] if 'longueurs' in missing
            else ["Une seule mesure au mètre pour l'échelle"] if 'echelle' in missing else ['Rien au mètre : recopie les cotes écrites sur le plan'] if copied else [])
    todo += [HEIGHT[type]] * ('hauteur' in missing) + ['Les portes et fenêtres : place, largeur et hauteur'] * ('ouvertures' in missing)
    return {'type': type, 'sources': chosen, 'ready': bool(answers['forme']), 'answers': answers, 'missing': missing, 'todo': todo,
            'offered': [{'id': s, 'label': SOURCES[s][0], 'answers': [q for q in QUESTIONS if q in SOURCES[s][1]]} for s in OFFERED[type]],
            'steps': [{'id': k, 'title': labels[k][0], 'hint': labels[k][1], 'useful': useful.get(k, True)} for k in FLOWS[type]]}


def detect(project, type):
    kinds, meta = {k for (k,) in project.db.execute('select kind from sources')}, project_meta(project)
    found = {'photos': 'photos_colmap' in kinds, 'scan': bool(glob.glob(os.path.join(project.folder, 'sources', 'scan', '*', 'passage-*'))), 'cadastre': any(k.startswith('cadastre') for k in kinds), 'relief': 'grille_alti' in kinds,
             'croquis': bool(meta.get('croquis')), 'manuel': bool((meta.get('guide') or {}).get('manual')), 'gabarit': True}
    return [s for s in OFFERED.get(type, OFFERED['piece']) if found.get(s)] or (['cadastre'] if type == 'maison' else [])


@rpc('guide.route')
def route(ctx, type, sources=None, walls=()):
    if sources is None: sources = detect(ctx.project, type) if ctx.project else ['cadastre'] if type == 'maison' else ['gabarit'] if type == 'vehicule' else []
    return parcours(type, sources, walls)

import itertools, json
L = ['SUPPORTED', 'UNSUPPORTED', 'CONTRADICTED']
A = dict(P01='U',P02='S',P03='U',P04='S',P05='S',P06='S',P07='U',P08='S',P09='C',P10='S')
B = dict(P01='U',P02='S',P03='U',P04='S',P05='S',P06='S',P07='S',P08='S',P09='U',P10='S')
V = dict(P01='U',P02='U',P03='U',P04='S',P05='S',P06='S',P07='U',P08='S',P09='C',P10='S')
ids = sorted(A)
def kappa(x, y):
    n = len(ids); po = sum(x[i]==y[i] for i in ids)/n
    pe = sum((sum(x[i]==c for i in ids)/n)*(sum(y[i]==c for i in ids)/n) for c in 'SUC')
    return po, pe, (po-pe)/(1-pe)
def macro_f1(gold, pred):
    f1s = {}
    for c in 'SUC':
        tp = sum(gold[i]==c and pred[i]==c for i in ids); fp = sum(gold[i]!=c and pred[i]==c for i in ids)
        fn = sum(gold[i]==c and pred[i]!=c for i in ids)
        if tp+fp+fn == 0: continue                      # class absent in both: excluded
        f1s[c] = 0.0 if tp == 0 else 2*tp/(2*tp+fp+fn)
    return sum(f1s.values())/len(f1s), f1s
def conf(gold, pred):
    return {g: {p: sum(gold[i]==g and pred[i]==p for i in ids) for p in 'SUC'} for g in 'SUC'}
po, pe, k = kappa(A, B)
print(f'A vs B: raw agreement {po:.2f} ({int(po*10)}/10), expected {pe:.2f}, Cohen kappa {k:.3f}')
print('disagreements:', [i for i in ids if A[i]!=B[i]])
for name, g in (('A', A), ('B', B)):
    acc = sum(g[i]==V[i] for i in ids)/10; mf, per = macro_f1(g, V)
    print(f'Verifier vs {name}: acc {acc:.2f}, macroF1 {mf:.3f} {per}')
agreed = [i for i in ids if A[i]==B[i]]
print(f'Verifier on {len(agreed)} agreed items: {sum(A[i]==V[i] for i in agreed)}/{len(agreed)}; wrong: {[i for i in agreed if A[i]!=V[i]]}')
print('\nScenario table (final label for P07, P09):')
for p7, p9 in itertools.product('SU', 'CU'):
    G = dict(A); G['P07'] = p7; G['P09'] = p9
    acc = sum(G[i]==V[i] for i in ids)/10; mf, per = macro_f1(G, V)
    print(f'  P07={p7} P09={p9}: verifier acc {acc:.1f}, macroF1 {mf:.3f}  per-class {{{", ".join(f"{c}:{v:.2f}" for c,v in per.items())}}}')
    print('    confusion rows=human, cols=verifier (S,U,C):', [[conf(G,V)[g][p] for p in 'SUC'] for g in 'SUC'])

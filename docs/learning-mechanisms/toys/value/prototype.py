"""Toy illustration of the proposed appraisal-coupled value learning (not an experiment).

Three ideas from the redesign, in the smallest form that runs:
1. The felt value of an outcome = innate values + values acquired from earlier experiences.
   Contributions add up exactly before the tanh bound, so every appraisal can be split
   into an innate part and the episodes that built each acquired value.
   Things that co-occur with valued outcomes acquire value themselves (evaluative conditioning).
2. What the personality learns from an event follows how it appraised that event:
   an outcome part caused by someone is credited to that someone, weighted by how much
   the personality currently cares about that part (attribution of responsibility).
3. Learning rate depends on whether the surprise confirms the current expectation
   (confirmation bias), and mood tilts the felt value (mood as momentum).

Fixed concept names are for readability only; the proposed design uses learned concept spaces.
"""
import numpy as np

F = ['connection', 'authorship', 'achievement', 'loss', 'novelty', 'autonomy', 'betrayal', 'music', 'friendA', 'others']
IDX = {f: i for i, f in enumerate(F)}
ENTITIES = {'friendA', 'others'}
# Innate values: identical for every personality, so all differences below come from experience.
INNATE = np.zeros(len(F)); INNATE[[IDX['achievement'], IDX['loss'], IDX['betrayal'], IDX['novelty']]] = [.5, -.6, -.4, .1]

def vec(**kw):
    x = np.zeros(len(F))
    for k, v in kw.items(): x[IDX[k]] = v
    return x

class Personality:
    def __init__(self, name, alpha=.25, kappa=.4, cooccur=.5, attribution=.3, beta=.3, decay=.002, value_based=True):
        self.name, self.alpha, self.kappa, self.cooccur, self.attribution, self.beta, self.decay = name, alpha, kappa, cooccur, attribution, beta, decay
        self.value_based = value_based  # False = control: appraisal from innate values only (3-F style fixed teacher)
        self.v = np.zeros(len(F)); self.mood = 0.
        self.ledger = []  # (episode, reason, delta vector); sums exactly to self.v

    def parts(self, outcome):
        """Per-feature contribution of the outcome to the felt value, split into innate and acquired."""
        return INNATE * outcome, (self.v * outcome if self.value_based else np.zeros(len(F)))

    def appraise(self, outcome):
        innate, acquired = self.parts(outcome); mood = self.beta * self.mood
        return dict(innate=float(innate.sum()), acquired=float(acquired.sum()), moodPart=mood,
                    felt=float(np.tanh(innate.sum() + acquired.sum() + mood)), components=innate + acquired)

    def expect(self, cue): return float(np.tanh(self.v @ cue))

    def _log(self, eid, reason, dv):
        self.v = self.v + dv; self.ledger.append((eid, reason, dv))

    def experience(self, eid, cue, outcome, caused_by=None):
        q = self.expect(cue); a = self.appraise(outcome); delta = a['felt'] - q
        confirms = q != 0 and np.sign(delta) == np.sign(q)
        rate = self.alpha * (1 + self.kappa if confirms else 1 - self.kappa)
        before = self.v.copy()
        self._log(eid, 'decay', -self.decay * self.v)
        # (a) The situation, and what co-occurred in the outcome, acquire the surprise in its felt value
        #     (evaluative conditioning: things paired with valued outcomes become valued themselves).
        self._log(eid, 'situation', rate * delta * cue)
        self._log(eid, 'co-occurrence', self.cooccur * rate * delta * np.where([f in ENTITIES for f in F], 0, outcome))
        # (b) Each caused outcome part is credited to whoever caused it, by how much it mattered to *this* personality.
        #     Like any prediction, it saturates: the entity moves toward what this event made it mean.
        for entity in sorted(set((caused_by or {}).values())):
            feats = [f for f, e in caused_by.items() if e == entity]
            cs = {f: a['components'][IDX[f]] for f in feats}; total = sum(cs.values()); held = self.v[IDX[entity]]
            if total:
                for f, c in cs.items(): self._log(eid, f'{entity} via {f}', self.attribution * np.tanh(total) * c / total * vec(**{entity: 1}))
            self._log(eid, f'{entity}: already expected', -self.attribution * np.tanh(held) * vec(**{entity: 1}))
        self.mood = .8 * self.mood + .2 * delta
        return dict(expected=q, delta=delta, rate=rate, dv=self.v - before, **a)

    def reasons(self, feature, episode_prefix=None):
        i = IDX[feature]; out = {}
        for eid, reason, dv in self.ledger:
            if episode_prefix and not eid.startswith(episode_prefix): continue
            key = (eid.rsplit('-', 1)[0], reason); out[key] = out.get(key, 0.) + dv[i]
        return {k: s for k, s in out.items() if abs(s) > 1e-3}

def history(kind):
    ev = []
    if kind == 'rescued-by-collaboration':
        ev += [('solo-fail', vec(autonomy=1, music=1), vec(autonomy=1, loss=.8, music=1), None)] * 6
        ev += [('collab-success', vec(connection=1, others=1, music=1), vec(connection=1, achievement=.9, others=1, music=1), {'connection': 'others'})] * 12
        ev += [('collab-ideas-dropped', vec(connection=1, others=1), vec(connection=1, authorship=-.8, achievement=.7, others=1), {'connection': 'others', 'authorship': 'others'})] * 4
    if kind == 'credit-taken':
        ev += [('credit-stolen', vec(connection=1, others=1, music=1), vec(connection=1, authorship=-1, betrayal=1, loss=.5, others=1, music=1), {'authorship': 'others', 'betrayal': 'others'})] * 8
        ev += [('solo-success', vec(autonomy=1, music=1), vec(autonomy=1, authorship=1, achievement=.9, music=1), None)] * 12
    if kind == 'explorer':
        ev += [('discovery', vec(autonomy=1), vec(autonomy=1, novelty=1, achievement=.3), None)] * 12
        ev += [('routine-success', vec(music=1), vec(novelty=-1, autonomy=-1, achievement=.6, music=1), None)] * 8
        ev += [('learned-from-others', vec(connection=1, others=1), vec(connection=1, novelty=.8, others=1), {'novelty': 'others'})] * 4
    return [(f'{k}-{i}', c, o, by) for i, (k, c, o, by) in enumerate(ev)]

# The same new event for everyone: a friend invites them to make a song; the song is well received,
# they learn a new technique from the friend, but almost none of their own ideas are used.
NEW = ('new-song-with-friendA-0', vec(connection=1, friendA=1, music=1),
       vec(connection=1, friendA=1, music=1, achievement=.8, authorship=-.8, novelty=.6),
       {'connection': 'friendA', 'authorship': 'friendA', 'novelty': 'friendA'})
INVITE_AGAIN = vec(connection=1, friendA=1, music=1)
KINDS = ['rescued-by-collaboration', 'credit-taken', 'explorer']

def run(value_based=True, show=True):
    rows = []
    for kind in KINDS:
        p = Personality(kind, value_based=value_based)
        for e in history(kind): p.experience(*e)
        values_before = p.v.copy(); invite_before = p.expect(INVITE_AGAIN)
        r = p.experience(*NEW); invite_after = p.expect(INVITE_AGAIN)
        rows.append(dict(p=p, values=values_before, r=r, invite=(invite_before, invite_after)))
        if show:
            comp = r['components']
            print(f'\n[{kind}] acquired values before: ' + ', '.join(f'{F[i]} {values_before[i]:+.2f}' for i in range(len(F)) if abs(values_before[i]) > .05))
            print(f'  felt value of the new event {r["felt"]:+.3f} = tanh(innate {r["innate"]:+.3f} + acquired {r["acquired"]:+.3f} + mood {r["moodPart"]:+.3f}); expected {r["expected"]:+.3f}, surprise {r["delta"]:+.3f}, learning rate {r["rate"]:.3f}')
            print('  what mattered: ' + ', '.join(f'{F[i]} {comp[i]:+.2f}' for i in np.argsort(-np.abs(comp))[:4]))
            print('  what friendA came to mean: ' + ', '.join(f'{reason} {s:+.3f}' for (_, reason), s in p.reasons('friendA', 'new-song').items()))
            print(f'  "friend A invites again": expected {invite_before:+.3f} -> {invite_after:+.3f}')
    return rows

if __name__ == '__main__':
    print('=== Proposed: appraisal uses innate + acquired values ===')
    rows = run(True)
    p = rows[1]['p']
    print('\nWhere "credit-taken" got its value for authorship:')
    for (ep, reason), s in p.reasons('authorship').items(): print(f'  {ep} ({reason}): {s:+.3f}')
    print('\n=== Control: appraisal uses innate values only (fixed teacher, as in 3-F) ===')
    ctrl = run(False)
    print('\nFelt value of the same outcome:   proposed', np.round([r['r']['felt'] for r in rows], 3), ' control', np.round([r['r']['felt'] for r in ctrl], 3))
    print('friendA value after the new event: proposed', np.round([r['p'].v[IDX['friendA']] for r in rows], 3), ' control', np.round([r['p'].v[IDX['friendA']] for r in ctrl], 3))

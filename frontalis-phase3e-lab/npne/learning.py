"""Explicit standard and exploratory appraisal views."""
from .core import experience
from .meaning import digest

def prepare_experience(row,representation,outcome_view='full'):
    if outcome_view not in ['full','facts8-neutral-context']:raise ValueError('Unknown outcome appraisal view.')
    e=experience(row,representation)
    if outcome_view=='facts8-neutral-context' and representation=='semantic44':
        e['observed'][8:]=[.5]*36
        e['sourceHash']=digest(dict(originalSourceHash=e['sourceHash'],view='facts8-neutral-context-v1'))
    return e

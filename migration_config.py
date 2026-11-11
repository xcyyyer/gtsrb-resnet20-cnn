"""Permit relocation of paths only; keep all experimental arguments fixed."""
PATH_KEYS = {'root','splits','out'}
def validate_relocation(old, new, enabled):
    differences={k:(old.get(k),new.get(k)) for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    if differences and (not enabled or set(differences)-PATH_KEYS):
        raise ValueError('Resume configuration changed beyond permitted relocation: '+repr(differences))
    return differences

import numpy as np

def cell2vector(value, *args):
    """
    Convert values of an individual cell/list into a vector.
    """
    if len(args) > 0:
        field = args[0]
    else:
        field = None

    try:
        # If it's a list/tuple/numpy array of objects/dicts
        if isinstance(value, (list, tuple, np.ndarray)):
            if field:
                res = []
                for val in value:
                    if hasattr(val, field):
                        res.append(getattr(val, field))
                    elif isinstance(val, dict) and field in val:
                        res.append(val[field])
                    else:
                        try:
                            res.append(val[field])
                        except:
                            res.append(None)
                return res
            else:
                return list(value)

        # If it is a single object/dict
        if field:
            if hasattr(value, field):
                attr = getattr(value, field)
                if isinstance(attr, (list, tuple)):
                    return list(attr)
                return attr
            elif isinstance(value, dict) and field in value:
                return value[field]
            else:
                return value[field]
        else:
            return value
    except Exception:
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return value[0]
        return value

import hashlib
import pickle

__all__ = ['hashn']

def hashn(generator):
    md5 = hashlib.md5()  # ignore
    for arg in generator:
        x = arg.data
        if hasattr(x, "cpu"):
            md5.update(x.cpu().numpy().data.tobytes())
        elif hasattr(x, "numpy"):
            md5.update(x.numpy().data.tobytes())
        elif hasattr(x, "data"):
            md5.update(x.data.tobytes())
        else:
            try:
                md5.update(x.encode("utf-8"))
            except:
                md5.update(str(x).encode("utf-8"))
    return md5.hexdigest()
from itertools import count, tee
from collections import deque

def ilen(it):
    consumeall = deque(maxlen=0).extend
    # Make a stateful counting iterator
    cnt = count()
    # zip it with the input iterator, then drain until input exhausted at C level
    consumeall(zip(it, cnt)) # cnt must be second zip arg to avoid advancing too far
    # Since count 0 based, the next value is the count
    return next(cnt)

def count_files(path, recursive=True, pattern='*'):
    if recursive:
        it = path.rglob(pattern)
    else:
        it = path.glob(pattern)
    return ilen(it)
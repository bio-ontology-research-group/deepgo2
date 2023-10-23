import numpy as np

AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
AANUM = len(AALETTER)
AAINDEX = dict()
for i in range(len(AALETTER)):
    AAINDEX[AALETTER[i]] = i + 1
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', '*'])
MAXLEN = 1000

def is_ok(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True

def to_tokens(seq):
    tokens = np.zeros((MAXLEN, ), dtype=np.float32)
    l = min(MAXLEN, len(seq))
    for i in range(l):
        tokens[i] = AAINDEX.get(seq[i], 21)
    return tokens

def to_onehot(seq, maxlen=MAXLEN, start=0):
    onehot = np.zeros((22, MAXLEN), dtype=np.float32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 21), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot

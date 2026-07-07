import numpy as np
from drive6 import Driver, obs

# shape algebra
def cw(s):
    g = [list(s[0:3]), list(s[3:6]), list(s[6:9])]
    n = [[g[2-c][r] for c in range(3)] for r in range(3)]
    return ''.join(''.join(row) for row in n)

S0 = '110011101'
ACHAIN = [S0]
_next = {S0: '010010111', '010010111': '101101111', '101101111': '011101010',
         '011101010': '010110011', '010110011': '111001101', '111001101': S0}
for _ in range(5):
    ACHAIN.append(_next[ACHAIN[-1]])

def classify(shape):
    """return (i, j) with shape == R^i(A^j(S0)), or None"""
    for j, base in enumerate(ACHAIN):
        x = base
        for i in range(4):
            if x == shape:
                return i, j
            x = cw(x)
    return None

COLORS = [14, 8, 12, 9]

def ops_needed(cur_shape, cur_color, tgt_shape, tgt_color):
    ci = classify(cur_shape); ti = classify(tgt_shape)
    if ci is None or ti is None:
        return None
    nA = (ti[1] - ci[1]) % 6
    nB = (ti[0] - ci[0]) % 4
    nC = (COLORS.index(tgt_color) - COLORS.index(cur_color)) % 4
    return nA, nB, nC

class Death(Exception):
    pass

class Exec(Driver):
    def step(self, a):
        prev_bar = self.o['bar']
        o = super().step(a)
        if o['bar'] > prev_bar + 20 and o['av'] == (50, 24):
            raise Death()
        return o

    def ensure_display(self, tgt_shape, tgt_color):
        guard = 0
        while (self.o['bl'], self.o['blc']) != (tgt_shape, [tgt_color]):
            guard += 1
            assert guard < 30
            need = ops_needed(self.o['bl'], self.o['blc'][0], tgt_shape, tgt_color)
            assert need, (self.o['bl'], self.o['blc'])
            nA, nB, nC = need
            # geographic order: C (middle), B (bottom), A (top)
            if nC:
                self.op_C()
            elif nB:
                self.op_B(n=nB)
            elif nA:
                self.op_A(n=nA)
            else:
                break

    def pocket_in(self):
        self.refuel_if_needed(floor=44)
        self.goto((15,49))
        self.step(1); self.step(1)   # (10,49) -> spring -> (25,49)
        assert self.o['av'] == (25,49), self.o['av']
        self.step(4); self.step(2)   # (30,54)

    def pocket_out(self):
        while self.o['av'] != (25,49):
            a = {(35,54):1, (30,54):1, (25,54):3}.get(self.o['av'])
            assert a, self.o['av']
            self.step(a)
        self.step(1)  # spring -> (20,39)

    def tr_open(self):
        f = np.asarray(self.c.frame())
        return int((f[35:41,53:60] == 8).sum()) == 0

    def phase1(self):
        if self.tr_open():
            return
        self.ensure_display('101110011', 8)   # TR = R^2(S0), color 8
        self.pocket_in()
        self.step(2)                           # press -> (35,54), 8s vanish
        assert self.o['av'] == (35,54) and self.tr_open(), self.o['av']
        self.step(1)                           # back to (30,54)
        self.pocket_out()

    def phase2(self):
        self.ensure_display('101001111', 9)   # BR = R^1(A^5(S0)), color 9
        assert self.tr_open()
        self.pocket_in()
        self.step(2)   # (35,54) through open TR
        self.step(2)   # (40,54)
        self.step(2)   # (45,54)
        assert self.o['av'] == (45,54), self.o['av']
        self.step(2)   # press into BR

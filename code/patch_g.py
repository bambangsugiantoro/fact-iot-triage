D = "/virtiofs/work/bambang/iot_exp/code/"
s = open(D + "core_v2.py").read()
reps = [('''                 window=4000, name="FACT (proposed)"):
        super().__init__(alpha)''', '''                 window=4000, name="FACT (proposed)", guard=0):
        super().__init__(alpha)
        self.guard = guard  # rare-class guard: alpha_c may not exceed alpha until a class has `guard` labels'''),
        ('''            self.nc = np.zeros(K)
            self.tX, self.ty, self.tw = [], [], []''', '''            self.nc = np.zeros(K)
            self.lc = np.zeros(K)
            self.tX, self.ty, self.tw = [], [], []'''),
        ('''                self.ac[c] = float(np.clip(self.ac[c] + self.gamma * upd, -0.2, 0.6))''', '''                self.lc[c] += m.sum()
                hi = self.alpha if (self.guard and self.lc[c] < self.guard) else 0.6
                self.ac[c] = float(np.clip(self.ac[c] + self.gamma * upd, -0.2, hi))''')]
for a, b in reps:
    assert s.count(a) == 1, a[:40]
    s = s.replace(a, b)
open(D + "core_v3.py", "w").write(s)
r = open(D + "run_v2.py").read()
a = 'methods = core if a.methods == "core" else allm'
assert r.count(a) == 1 and r.count("from core_v2 import") == 1
r = r.replace("from core_v2 import", "from core_v3 import").replace(a, '''if a.methods == "guard":
    methods = [FACT(al, eps=a.eps, guard=30, name="FACT-G (rare-class guard)")]
else:
    methods = core if a.methods == "core" else allm''')
open(D + "run_v3.py", "w").write(r)
print("patched_g ok")

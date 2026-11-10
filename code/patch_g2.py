D = "/virtiofs/work/bambang/iot_exp/code/"
s = open(D + "core_v3.py").read()
a = '''            else:
                m = ty == c
                q[c] = q_weighted(self.tscore[m], tw[m], 1 - a, w_test=1.0) if m.any() else np.inf'''
b = '''            else:
                m = ty == c
                if self.guard and len(sc) > 0:
                    # rare-class guard: pool the few source scores with the labelled deployment scores;
                    # deployment weights are Hajek-normalised to their count so that the finite-sample
                    # correction reflects the real number of labelled flows
                    wd = tw[m] * (m.sum() / tw[m].sum()) if m.any() else np.zeros(0)
                    q[c] = q_weighted(np.r_[sc, self.tscore[m]], np.r_[np.ones(len(sc)), wd], 1 - a, w_test=1.0)
                else:
                    q[c] = q_weighted(self.tscore[m], tw[m], 1 - a, w_test=1.0) if m.any() else np.inf'''
assert s.count(a) == 1
s = s.replace(a, b)
open(D + "core_v4.py", "w").write(s)
r = open(D + "run_v3.py").read()
assert r.count("from core_v3 import") == 1
open(D + "run_v4.py", "w").write(r.replace("from core_v3 import", "from core_v4 import"))
print("patched_g2 ok")

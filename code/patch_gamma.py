D = "/virtiofs/work/bambang/iot_exp/code/"
r = open(D + "run_v3.py").read()
a = 'ap.add_argument("--label_noise", type=float, default=0.0)'
assert r.count(a) == 1
r = r.replace(a, a + '\nap.add_argument("--gamma", type=float, default=0.05)')
b = 'if a.methods == "guard":'
assert r.count(b) == 1
r = r.replace(b, '''if a.methods == "factonly":
    methods = [FACT(al, eps=a.eps, gamma=a.gamma)]
elif a.methods == "guard":''')
open(D + "run_v3.py", "w").write(r)
print("patched_gamma ok")

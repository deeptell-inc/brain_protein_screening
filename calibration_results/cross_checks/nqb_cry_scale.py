import numpy as np
exec(open(__file__.replace("nqb_cry_scale","nqb_cry_premise")).read().split("cases = [")[0])
print(f"{'case':40s} IPC")
for lab,J,r in [("J only +12.6 MHz, D=0",12.6,None),("J only +1.26 MHz, D=0",1.26,None),
                ("D at r=2.2 nm (ET-limited max, Efimova)",0.0,2.2),("D at r=2.5 nm",0.0,2.5),
                ("D at r=3.5 nm (Efimova 'weak enough')",0.0,3.5),("D at r=5.0 nm",0.0,5.0)]:
    Dm = 0.0 if r is None else -77.91/r**3
    m,sd = ipc(H_with(J,Dm,"z"))
    print(f"  {lab:40s} D={Dm:7.2f} MHz  IPC = {m:.3f} +/- {sd:.3f}", flush=True)

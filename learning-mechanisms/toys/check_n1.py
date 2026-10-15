import numpy as np
# N1: self-including target, gain 1
a, al = 0.5, 0.25*1.4*1.5
for d in [0.0, 0.002]:
    v=0.; out=[]
    for t in range(1,100001):
        v += al*(np.tanh(a+v)-np.tanh(v)) - d*v
        if t in (10,100,1000,10000,100000): out.append(round(v,3))
    print('decay',d,'v at t=10..1e5',out)
# N2: contraction forgets history: two personalities with different v0, same new experiences
for g in [0.0,0.5,0.9]:
    v1,v2=1.5,-1.5
    for t in range(8): 
        v1 += 0.3*((a+g*v1)-v1); v2 += 0.3*((a+g*v2)-v2)
    print('gain',g,'diff after 8 new exps',round(v1-v2,3),'of 3.0')
# N4: bounded positive feedback g>1 -> bistable
def run(v,u,g,n=200):
    for t in range(n): v += 0.2*(-v+g*np.tanh(v)+u)
    return v
for g in [0.8,1.5,2.5]:
    print('g',g,'from +1 with counter u=-0.2:',round(run(1.0,-0.2,g),2),' u=-0.6:',round(run(1.0,-0.6,g),2), ' from -1 u=-0.2:',round(run(-1.0,-0.2,g),2))

import unittest
import numpy as np
from sequential import estimate, centered, solve_theta


class SequentialTests(unittest.TestCase):
    def setUp(self):
        self.model={'groups':['s'],'points':{}}
        for i in range(30):
            q=np.array([1500+i*20,1000+(i%4)*20.])
            a=np.array([14.4,0.]);b=np.array([80.+(i-15)*.1,(i%4-1.5)*.3]);c=np.array([(i-15)*.02,0.])
            self.model['points'][f'QR{i//10}:{i}']={'reference':q.tolist(),'coefficients':np.array([[.4,-.1],a,b,c]).tolist()}

    def observation(self,y,t):
        return {k:(np.array(p['reference'])+np.array(p['coefficients'])[0]+
                   np.array(p['coefficients'])[1]*y+np.array(p['coefficients'])[2]*t+
                   np.array(p['coefficients'])[3]*t*t).tolist() for k,p in self.model['points'].items()}

    def test_mixed_recovery(self):
        for y in [-1,.05,1.5]:
            for t in [-.8,-.01,.3,.9]:
                r=estimate(self.model,self.observation(y,t),'s')
                self.assertAlmostEqual(r['theta_deg'],t,places=7)
                self.assertAlmostEqual(r['y_mm'],y,places=7)

    def test_theta_ignores_arbitrary_mean(self):
        obs=self.observation(.5,.2);shifted={k:(np.array(v)+[300,-50]).tolist() for k,v in obs.items()}
        self.assertAlmostEqual(estimate(self.model,obs,'s')['theta_deg'],estimate(self.model,shifted,'s')['theta_deg'],places=8)

    def test_missing_points_recenter_same_subset(self):
        obs=self.observation(.3,-.4)
        for k in list(obs)[::6]:del obs[k]
        r=estimate(self.model,obs,'s');self.assertAlmostEqual(r['theta_deg'],-.4,places=7)
        self.assertAlmostEqual(r['y_mm'],.3,places=7)

    def test_reject_unobservable_theta(self):
        with self.assertRaises(ValueError):solve_theta(np.zeros((30,2)),np.ones((30,2)),np.zeros((30,2)))

    def test_reject_insufficient(self):
        with self.assertRaises(ValueError):estimate(self.model,{})


if __name__=='__main__':unittest.main()

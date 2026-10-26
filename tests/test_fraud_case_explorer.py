"""Descriptive evidence must not imply motifs unsupported by directed data."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('explorer', Path(__file__).resolve().parents[1] / 'src/visualization/fraud_case_explorer.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def edge(i, a, b, t=0, fraud=0, currency='USD'):
    return dict(id=str(i), src=a, dst=b, seconds=t, fraud=fraud, amount=10,
                currency=currency, bank='Key_Bank', format='Wire', t=str(t))


class EvidenceTests(unittest.TestCase):
    def test_fan_requires_distinct_peers_and_excludes_self(self):
        focal=edge(0,'a','z',3,1)
        repeated=[edge(i,'a','z',i) for i in range(1,3)]
        result=m.analyze_case(focal,[focal,*repeated,edge(4,'a','a')])
        self.assertFalse(result['motifs'])
        result=m.analyze_case(focal,[focal,edge(5,'a','b'),edge(6,'a','c')])
        self.assertEqual(result['motifs'][0]['kind'],'Fan-out')
        self.assertEqual(result['motifs'][0]['peers'],3)

    def test_direction_and_chronology_are_separate(self):
        focal=edge(0,'a','b',0,1)
        path=[focal,edge(1,'b','c',3),edge(2,'c','d',2),edge(3,'d','a',4)]
        cycles,capped=m.cycle_paths(focal,path)
        self.assertEqual(len(cycles),1)
        self.assertFalse(m.chronological_cycle(cycles[0]))
        path[1]['seconds']=1
        self.assertTrue(m.chronological_cycle(path))
        path[1]['seconds']=0
        self.assertFalse(m.chronological_cycle(path))
        self.assertFalse(m.cycle_paths(focal,[focal,edge(9,'a','c')])[0])

    def test_self_transfer_not_a_cycle(self):
        focal=edge(0,'a','a',0,1)
        self.assertEqual(m.cycle_paths(focal,[focal]),([],False))

    def test_prior_excludes_future_and_simultaneous_and_baseline_currency(self):
        focal=edge(2,'a','b',100,1)
        rows=[edge(0,'a','c',0),edge(1,'a','d',1,currency='EUR'),focal,
              edge(3,'b','a',100),edge(4,'b','a',101)]
        data=m.build_payload(rows);c=data['cases'][0]
        self.assertNotIn('3',c['views']['1-prior']['edges'])
        self.assertNotIn('4',c['views']['1-prior']['edges'])
        self.assertIn('4',c['views']['1-around']['edges'])
        self.assertEqual(c['baseline']['peerCount'],1)
        self.assertEqual(c['baseline']['ownCount'],1)

    def test_display_cap_does_not_change_full_fan_counts(self):
        focal=edge(0,'a','b',1000,1)
        rows=[focal]+[edge(i,'a',str(i),i) for i in range(1,120)]
        result=m.analyze_case(focal,rows,display_limit=20)
        self.assertEqual(len(result['edges']),20)
        self.assertEqual(result['motifs'][0]['peers'],120)
        self.assertTrue(result['displayLimited'])
        self.assertIn(focal['id'],result['edges'])

    def test_cycle_work_cap_disclosed(self):
        focal=edge(0,'a','b',0,1)
        _,capped=m.cycle_paths(focal,[focal,edge(1,'b','c'),edge(2,'c','a')],budget=1)
        self.assertTrue(capped)

if __name__=='__main__':
    unittest.main()

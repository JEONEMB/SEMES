"""Exercise the actual display functions without image loads or report writes."""
import ast,html,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'reference_line_methods_108_20261030'
def function(filename,name,namespace=None):
    tree=ast.parse((BASE/'tools'/filename).read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),ns)
    return ns[name]

class DisplayContract(unittest.TestCase):
    def test_selected_segment_draws_clipped_pieces(self):
        draw=function('make_figures.py','display_segment_rows')
        source=[0,0,100,0,0,100,20,1,'selected']
        rejected=[0,3,100,3,0,100,20,1,'excluded:not_in_consensus_line']
        data={'intermediates':{'segments':[source,rejected],'visible_segment_pieces':[
            {'segment_index':0,'endpoints_xy':[[10,0],[30,0]],'state':'selected'},
            {'segment_index':0,'endpoints_xy':[[70,0],[90,0]],'state':'selected'}]}}
        rows=draw(data)
        self.assertEqual([r[:4] for r in rows if r[-1]=='selected'],[[10,0,30,0],[70,0,90,0]])
        self.assertIn(rejected,rows)
        self.assertEqual(data['intermediates']['segments'][0],source)
    def test_legacy_segment_records_are_not_reconstructed(self):
        draw=function('make_figures.py','display_segment_rows')
        rows=[[0,0,100,0,0,100,20,1,'selected']]
        self.assertEqual(draw({'intermediates':{'segments':rows}}),rows)
    def test_legacy_settings_are_marked_unrecorded(self):
        table=function('make_report.py','settings_table',{'E_IDS':['E04'],'Q_IDS':[],
            'lm':{'records':[{'method_id':'E04'}]},'e':html.escape,'json':json})
        self.assertIn('기록 없음',table())
    def test_current_settings_table_uses_stored_scope(self):
        settings={'applied':{'target':{},'common_fit':{}},'inactive':{'method':{'sigma_scale':{'reason':'inactive'}}},
                  'reporting_only':{},'fixed_modes':{}}
        table=function('make_report.py','settings_table',{'E_IDS':['E04'],'Q_IDS':[],
            'lm':{'records':[{'method_id':'E04','target_id':'upper_structure','settings':settings}]},'e':html.escape,'json':json})
        output=table()
        self.assertIn('Huber: 적용',output)
        self.assertIn('band: 미사용',output)
        self.assertIn('sigma_scale',output)

if __name__=='__main__':unittest.main()

"""No API, model downloads or environment secrets are needed for these tests."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import chat_rag_roles_prompt_only as candidate
from rag_dashboard_backend import evaluation_cards

class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.question="ما الذي توضحه سياسة الخصوصية؟"
        self.parts=[{"id":"P1","topic":"question","query":self.question}]
        self.sources={"pdpl:12:ar":{"document_id":"pdpl","article_number":12,"language":"ar",
            "text":"توضح سياسة الخصوصية الغرض من جمع البيانات الشخصية وكيفية معالجتها."}}
        self.segments=candidate.engine.v2.source_segments(self.sources)
        self.eid=next(iter(self.segments))
        self.raw={"claims":[{"text":self.sources["pdpl:12:ar"]["text"],"evidence_segment_ids":[self.eid]}],
            "question_parts":[{"part_id":"P1","part":self.question,"coverage":"covered",
                "claim_numbers":[1],"missing_information":""}],"requirement_coverage":[]}
    def validate(self,raw,segments=None,units=None):
        return candidate.validate(json.dumps(raw,ensure_ascii=False),self.parts,self.sources,
            self.segments if segments is None else segments,[] if units is None else units,"ar")
    def test_no_actor_output_gate(self):
        schema=candidate.output_schema(self.parts,self.segments,[])
        self.assertNotIn("actors",schema["properties"])
        self.assertNotIn("actor_ids",schema["$defs"]["DraftClaim"]["properties"])
        answer,*_=self.validate(self.raw)
        self.assertEqual(answer.status,"answered")
    def test_unknown_evidence_rejected(self):
        raw=copy.deepcopy(self.raw);raw["claims"][0]["evidence_segment_ids"]=["E999999"]
        with self.assertRaises((ValueError,candidate.engine.base.ResponseValidationError)):self.validate(raw)
    def test_modified_quote_rejected(self):
        segments=copy.deepcopy(self.segments);segments[self.eid]["text"]+=" changed"
        with self.assertRaises((ValueError,candidate.engine.base.ResponseValidationError)):self.validate(self.raw,segments)
    def test_changed_question_scope_rejected(self):
        raw=copy.deepcopy(self.raw);raw["question_parts"][0]["part"]="A different question"
        with self.assertRaises(ValueError):self.validate(raw)
    def test_requirement_evidence_still_required(self):
        raw=copy.deepcopy(self.raw)
        raw["requirement_coverage"]=[{"requirement_id":"R001","coverage":"covered",
            "claim_numbers":[1],"part_ids":["P1"],"reason":""}]
        units=[{"requirement_id":"R001","source_available":True,"required_evidence_ids":["E999999"]}]
        with self.assertRaises(ValueError):self.validate(raw,units=units)
    def test_failure_and_quality_rates_are_distinct(self):
        summary={"questions":20,"full_four_criteria_answers":13,"correct_abstentions":4,
            "execution_errors":2,"execution_success_rate":.9,"unanswerable":4,
            "accepted_answerable_answers":14}
        cards=evaluation_cards(summary)
        self.assertEqual(cards["complete"],17)
        self.assertEqual(cards["execution_success"],18)
        self.assertNotEqual(cards["complete_rate"],cards["execution_rate"])
if __name__=="__main__":unittest.main()

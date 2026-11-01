import unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import gap_analyzer as g
class GapChecks(unittest.TestCase):
 def setUp(self):
  self.item=g.read_json(g.CHECKLIST)["items"][1]
 def test_saved_covered_needs_quote(self):
  saved={"requirement_id":self.item["id"],"status":"covered","checks":[{"check_id":self.item["checks"][0]["id"],"status":"covered","evidence":[]}]}
  self.assertFalse(g.validate_saved(self.item,saved,"Policy"))
 def test_fresh_invented_quote_rejected(self):
  raw={"checks":[{"check_id":self.item["checks"][0]["id"],"status":"covered","reason":"test","quote":"invented"}]}
  with self.assertRaises(ValueError):g.validate_results(raw,self.item["checks"],"Policy",["Policy"])
 def test_payload_contains_legal_text(self):
  index=g.build_legal_index(g.read_json(g.LEGAL));old=g.ask_gemini;seen={}
  def fake(client,payload):
   seen.update(payload)
   return {"checks":[{"check_id":c["id"],"status":"needs_review","reason":"test","quote":""} for c in payload["checks"]]}
  try:
   g.ask_gemini=fake
   result=g.analyze_requirement(None,self.item,"Policy",["Policy"],index,candidate_evidence=[{"paragraph_text":"Policy"}])
  finally:g.ask_gemini=old
  self.assertTrue(seen["legal_evidence"]);self.assertTrue(seen["candidate_policy_evidence"]);self.assertTrue(g.validate_saved(self.item,result,"Policy"))
if __name__=="__main__":unittest.main()

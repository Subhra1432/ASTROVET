import unittest
from astrovet.ids import (
    resolve_candidate_identity,
    make_canonical_candidate_id,
    parse_candidate_string,
    CandidateIdentity,
)


class TestCandidateIdentity(unittest.TestCase):
    def test_canonical_id_format(self):
        cid = make_canonical_candidate_id(390651552, 1, "23")
        self.assertEqual(cid, "390651552-1-S23")

        cid_multi = make_canonical_candidate_id(123456789, 2, "S1-3")
        self.assertEqual(cid_multi, "123456789-2-S1-3")

    def test_parse_candidate_string(self):
        tic, plnt, sec = parse_candidate_string("390651552-1-S23")
        self.assertEqual((tic, plnt, sec), (390651552, 1, "23"))

        tic2, plnt2, sec2 = parse_candidate_string("TIC 000390651552 2 S14")
        self.assertEqual((tic2, plnt2, sec2), (390651552, 2, "14"))

    def test_reject_naked_tic(self):
        # TIC ID alone cannot uniquely identify a transit event
        with self.assertRaises(ValueError):
            parse_candidate_string("390651552")

        with self.assertRaises(ValueError):
            parse_candidate_string("TIC 390651552")

    def test_resolve_candidate_identity(self):
        ident = resolve_candidate_identity(candidate_id="390651552-1-S23")
        self.assertIsInstance(ident, CandidateIdentity)
        self.assertEqual(ident.tic_id, 390651552)
        self.assertEqual(ident.planet_num, 1)
        self.assertEqual(ident.sector_run, "23")
        self.assertEqual(ident.candidate_id, "390651552-1-S23")


if __name__ == "__main__":
    unittest.main()

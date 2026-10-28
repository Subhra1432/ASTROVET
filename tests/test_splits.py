import unittest
from astrovet.evaluation.splits import make_host_grouped_split


class TestHostGroupedSplits(unittest.TestCase):
    def test_zero_leakage_across_partitions(self):
        # 10 host stars, multiple candidates per star
        records = []
        for tic in range(100, 110):
            for plnt in range(1, 3):
                records.append({
                    "candidate_id": f"{tic}-{plnt}-S1",
                    "tic_id": tic,
                    "planet_num": plnt,
                    "sector_run": "1",
                })

        splits = make_host_grouped_split(
            records,
            train_frac=0.60,
            val_frac=0.15,
            cal_frac=0.10,
            test_frac=0.15,
            random_seed=42,
        )

        train_tics = {r["tic_id"] for r in splits["train"]}
        val_tics = {r["tic_id"] for r in splits["val"]}
        cal_tics = {r["tic_id"] for r in splits["cal"]}
        test_tics = {r["tic_id"] for r in splits["test"]}

        # Strictly group disjoint
        self.assertEqual(len(train_tics.intersection(val_tics)), 0)
        self.assertEqual(len(train_tics.intersection(cal_tics)), 0)
        self.assertEqual(len(train_tics.intersection(test_tics)), 0)
        self.assertEqual(len(val_tics.intersection(test_tics)), 0)
        self.assertEqual(len(cal_tics.intersection(test_tics)), 0)

        # All records accounted for
        total_split = sum(len(s) for s in splits.values())
        self.assertEqual(total_split, len(records))


if __name__ == "__main__":
    unittest.main()

import unittest
from rpm_shared_capacity import capacity_decision


class SharedCapacityTests(unittest.TestCase):
    def decision(self,**changes):
        values=dict(logical_cpus=256,physical_cpus=128,busy_logical_cpus=5.,load_one=5.,
            total_memory_gib=503.,available_memory_gib=478.,workers=8)
        values.update(changes)
        return capacity_decision(**values)

    def test_other_active_compute_is_allowed_with_headroom(self):
        self.assertTrue(self.decision()["allowed"])

    def test_cpu_pressure_holds_new_starts(self):
        self.assertFalse(self.decision(busy_logical_cpus=110.)["allowed"])

    def test_large_run_queue_holds_even_when_sampled_cpu_idle(self):
        self.assertFalse(self.decision(load_one=120.)["allowed"])

    def test_memory_reserve_is_preserved(self):
        self.assertFalse(self.decision(available_memory_gib=60.)["allowed"])

    def test_quarter_of_physical_cores_remains_reserved(self):
        self.assertEqual(self.decision()["reserved_physical_cores"],32)

    def test_capacity_recovery_allows_a_later_retry(self):
        self.assertFalse(self.decision(busy_logical_cpus=110.)["allowed"])
        self.assertTrue(self.decision(busy_logical_cpus=20.)["allowed"])

    def test_invalid_worker_count_is_rejected(self):
        with self.assertRaises(ValueError):self.decision(workers=0)


if __name__=="__main__":unittest.main()

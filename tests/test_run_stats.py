"""Stage bookkeeping of run_stats with a fake memory reader, and the real reader."""

import time

import pytest

from tnx_omop.util.run_stats import NullMemoryReader, RunStats, default_reader


class FakeReader:
    """Returns the queued values in turn, then the last one."""

    counter = "fake"

    def __init__(self, values: list[int | None], peak: int | None = 99) -> None:
        self.values = values
        self.peak = peak

    def current(self) -> int | None:
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]

    def process_peak(self) -> int | None:
        return self.peak


class TestStage:
    def test_records_time_and_peak_of_each_stage(self):
        reader = FakeReader([10, 50, 30, 5, 7])
        with RunStats(reader, interval=3600) as stats:
            with stats.stage("a"):
                time.sleep(0.01)
            with stats.stage("b"):
                pass

        a, b = stats.stages["a"], stats.stages["b"]
        assert list(stats.stages) == ["a", "b"]
        assert a.seconds >= 0.01
        # stage start and end are sampled even when no timer tick happens
        assert a.peak_memory_bytes == 50
        assert b.peak_memory_bytes == 30
        assert stats.process_peak_bytes == 99
        assert stats.counter == "fake"

    def test_sampling_thread_raises_the_peak(self):
        reader = FakeReader([1, 1000, 2])
        with RunStats(reader, interval=0.001) as stats:
            with stats.stage("a"):
                time.sleep(0.05)

        assert stats.stages["a"].peak_memory_bytes == 1000

    def test_unknown_memory_is_none(self):
        with RunStats(NullMemoryReader()) as stats:
            with stats.stage("a"):
                pass

        assert stats.stages["a"].peak_memory_bytes is None
        assert stats.process_peak_bytes is None
        assert stats.counter == "unknown"

    def test_failing_stage_is_recorded_and_not_active(self):
        with RunStats(FakeReader([4])) as stats:
            with pytest.raises(ValueError):
                with stats.stage("a"):
                    raise ValueError("boom")
            with stats.stage("b"):
                pass

        assert stats.stages["a"].peak_memory_bytes == 4
        assert stats.stages["a"].seconds > 0

    def test_nested_stages_both_see_the_peak(self):
        with RunStats(FakeReader([1, 8, 2]), interval=3600) as stats:
            with stats.stage("outer"):
                with stats.stage("inner"):
                    pass

        assert stats.stages["outer"].peak_memory_bytes == 8
        assert stats.stages["inner"].peak_memory_bytes == 8

    def test_repeated_name_adds_up_time_and_keeps_max(self):
        with RunStats(FakeReader([5, 9, 3, 1]), interval=3600) as stats:
            with stats.stage("a"):
                time.sleep(0.01)
            with stats.stage("a"):
                time.sleep(0.01)

        assert stats.stages["a"].seconds >= 0.02
        assert stats.stages["a"].peak_memory_bytes == 9


class TestDefaultReader:
    def test_process_peak_is_positive(self):
        reader = default_reader()

        assert reader.counter != "unknown"
        assert reader.process_peak() > 0

    def test_current_is_positive_where_supported(self):
        reader = default_reader()
        current = reader.current()

        assert current is None or current > 0
        assert current is not None or reader.counter == "ru_maxrss"

    def test_current_does_not_exceed_peak_much(self):
        reader = default_reader()
        current, peak = reader.current(), reader.process_peak()

        if current is not None:
            assert current <= peak * 1.5

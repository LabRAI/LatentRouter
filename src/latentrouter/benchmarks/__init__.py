from latentrouter.benchmarks import mmr_bench, vl_routerbench

BENCHMARKS = {"mmr_bench": mmr_bench, "vl_routerbench": vl_routerbench}

__all__ = ["BENCHMARKS", "mmr_bench", "vl_routerbench"]

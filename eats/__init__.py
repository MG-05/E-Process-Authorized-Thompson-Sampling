"""The frozen e-process-authorized Thompson sampling policy and open ramp."""
__all__ = ['run_e_ats', 'run_open_ramp']

def __getattr__(name):
    if name in __all__:
        from eats.algorithms.e_ats import run_e_ats_full_budget_ramp, run_e_ats_full_budget_ramp_no_ce
        return run_e_ats_full_budget_ramp if name == 'run_e_ats' else run_e_ats_full_budget_ramp_no_ce
    raise AttributeError(name)

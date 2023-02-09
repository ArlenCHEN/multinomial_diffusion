from easydict import EasyDict

eval_cfg = EasyDict()

eval_cfg.is_jump_schedule = True
if eval_cfg.is_jump_schedule:
  eval_cfg.jump_schedule = EasyDict()
  eval_cfg.jump_schedule.t_T = 250
  eval_cfg.jump_schedule.n_sample = 1
  eval_cfg.jump_schedule.jump_length = 10
  eval_cfg.jump_schedule.jump_n_sample = 10
  eval_cfg.jump_schedule.inpa_inj_sched_prev = True
  eval_cfg.jump_schedule.inpa_inj_sched_prev_cumnoise = False
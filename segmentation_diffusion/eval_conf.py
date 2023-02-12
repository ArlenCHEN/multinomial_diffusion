from easydict import EasyDict

eval_cfg = EasyDict()

eval_cfg.is_jump_schedule = True
if eval_cfg.is_jump_schedule:
  eval_cfg.jump_schedule = EasyDict()
  eval_cfg.jump_schedule.t_T = 4000
  eval_cfg.jump_schedule.n_sample = 1
  eval_cfg.jump_schedule.jump_length = 1 # (j) how many steps the model looks back  
  eval_cfg.jump_schedule.jump_n_sample = 1 # (r) how many resampling circles are performed
  eval_cfg.jump_schedule.inpa_inj_sched_prev = True
  eval_cfg.jump_schedule.inpa_inj_sched_prev_cumnoise = False
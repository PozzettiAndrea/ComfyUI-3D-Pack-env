# stable-diffusion-2-1-base scheduler

`stabilityai/stable-diffusion-2-1-base` is GATED: an anonymous request returns
401 RepositoryNotFound, so `DDIMScheduler.from_pretrained("stabilityai/...")`
now fails at runtime for every CRM family. The three CRM `model.py` files load
this directory instead.

Values are the published SD 2.1 *base* schedule, taken from the ungated mirror
`flax/stable-diffusion-2-1-base` (scheduler/scheduler_config.json).

Note `prediction_type: "epsilon"`. The 768px `stable-diffusion-2-1` uses
`v_prediction` -- substituting that model's scheduler here would not error, it
would just quietly produce a wrong denoising schedule.

`_class_name` says DDIMScheduler because that is the class that loads it; the
mirror labels it FlaxPNDMScheduler, and the shared parameters are identical.

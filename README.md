### Notes

- Send `{ "input": { "warmup": true } } first to warm up the model before you actually sending your prompt. Be quick though you only have a few seconds until it has
to cold start again.
- The model prompt inference engine is still too dumb.
- Available samplers:
```
mapping = {
        "pndm": _mk(PNDMScheduler),
        "plms": _mk(PNDMScheduler),
        "lms": _mk(LMSDiscreteScheduler),
        "heun": _mk(HeunDiscreteScheduler),
        "euler": _mk(EulerDiscCls),
        "euler_a": _mk(EulerAncCls),
        "euler_ancestral": _mk(EulerAncCls),
        "ddim": _mk(DDIMScheduler),
        "kdpm2": _mk(KDPM2DiscreteScheduler), "dpm2": _mk(KDPM2DiscreteScheduler),
        "kdpm2_a": _mk(KDPM2AncestralDiscreteScheduler), "kdpm2_ancestral": _mk(KDPM2AncestralDiscreteScheduler), "dpm2_a": _mk(KDPM2AncestralDiscreteScheduler),
        "deis": _mk(DEISMultistepScheduler),
        "ddpm": _mk(DDPMScheduler),
        "dpmpp_2m": _mk(DPMSolverMultistepScheduler),
        "dpmpp_2m_sde": _mk(DPMSolverMultistepScheduler),
        "dpmpp_sde": _mk(DPMSolverSinglestepScheduler),
        "dpmsolver": _mk(DPMSolverSinglestepScheduler),
        "uni_pc": _mk(UniPCMultistepScheduler),
        "flowmatch_euler": _mk(FMEDS),
    }
```
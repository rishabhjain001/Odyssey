# Odyssey

Odyssey is a world-model reinforcement learning agent, in the style of
DreamerV2, that learns to drive in the Gymnasium `CarRacing-v2` environment
directly from raw pixel frames. It compresses observations into a discrete
latent space, learns the environment dynamics in that latent space (the
"world model"), and trains an actor-critic policy entirely inside imagined
latent rollouts, so policy improvement consumes no real environment steps.
Odyssey is compared against a model-free Proximal Policy Optimization (PPO)
baseline to study sample efficiency.

## Motivation

Model-free agents often require large numbers of environment interactions to
learn control from pixels. World-model methods aim to reduce that cost by
learning a predictive model of the environment and training behavior inside
the model. Odyssey is a focused, readable implementation of this idea for a
single continuous-control driving task, intended for study and comparison
rather than as a production system.

## Scope and Staging

The project is delivered in deliberate stages, and a core discipline governs
the order: no policy training or baseline work begins until frame
reconstruction is visibly working.

1. Minimum Viable Product (MVP): grayscale perception, a discrete categorical
   latent, frame reconstruction, reconstruction-loss tracking, and a minimal
   Streamlit dashboard.
2. A reconstruction gate enforces the staging discipline. The
   `Reconstruction_Confirmed_Condition` keeps behavior-model training and the
   PPO baseline disabled until reconstruction quality is confirmed.
3. Later stages add RGB perception, longer imagination sequences, imagined
   actor-critic training, the PPO comparison, and a PDF writeup.

Scope is driven by configuration. Channel count and sequence length flow from
a single config file, so moving from grayscale to RGB requires no code change.

## Project Layout

```
odyssey/
├── train.py            # single orchestrating entry point (added in a later stage)
├── requirements.txt    # Python dependencies
├── .env.example        # documented environment variables
├── README.md           # this file
├── agent/              # Encoder, RSSM, Decoder, Reward/Discount, Actor, Critic
├── training/           # world-model + behavior training loops, gate, pipeline
├── baseline/           # PPO baseline (Stable-Baselines3)
├── dashboard/          # Streamlit app
├── configs/            # YAML config files
├── scripts/            # helper scripts
└── report/             # ReportLab writeup generator
```

## Getting Started

```bash
pip install -r requirements.txt
cp .env.example .env   # then adjust values as needed
```

The training entry point and configuration files are introduced in later
implementation stages.

## Limitations

- Odyssey targets a single environment (`CarRacing-v2`) and is not a general
  world-model framework.
- Results depend on hyperparameters and available compute; the defaults favor
  readability and reproducibility over peak performance.
- The PPO baseline relies on Stable-Baselines3 rather than a custom
  implementation, so baseline behavior reflects that library's choices.
- Sample-efficiency comparisons are indicative for this task and seed range
  and should not be read as general claims about the methods.
- The MVP operates on grayscale frames; RGB perception is a later stage.

## References

- Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., &
  Davidson, J. (2018). *Learning Latent Dynamics for Planning from Pixels*
  (PlaNet / RSSM). arXiv:1811.04551.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019). *Dream to Control:
  Learning Behaviors by Latent Imagination* (Dreamer). arXiv:1912.01603.
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2020). *Mastering Atari
  with Discrete World Models* (DreamerV2). arXiv:2010.02193.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms* (PPO). arXiv:1707.06347.

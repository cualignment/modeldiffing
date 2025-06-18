# Design
├── README.md
├── docs/
│   └── design.md
├── configs/
│   ├── default.yaml
│   └── ppo_llama.yaml
├── src/
│   ├── models/

crlt + f todos
design choices that i would like to experiement with: the architecture of the critic. last token/vs some average over all tokens?

right now in the value head we only train the head. would love to see how the performance changes if we train the whole thing

when computing the rewards should use exp average???
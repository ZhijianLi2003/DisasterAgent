<p align="center">
  <img src="assets/icon.png" alt="DisasterAgent" width="150"/>
</p>

<h1 align="center"><strong>DisasterAgent: An Evidence-Driven Agent for Verifiable Disaster Report Generation from Remote Sensing Imagery</strong></h1>

<p align="center">
  <strong><a href="https://zhijianli2003.github.io/">Zhijian Li</a></strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Jiawei Song</strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Chao Ren</strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Yidan Zhang</strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Peijin Wang</strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Xian Sun</strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Kun Fu</strong><br/>
  Aerospace Information Research Institute, Chinese Academy of Sciences<br/>
  University of Chinese Academy of Sciences<br/>
  Key Laboratory of Target Cognition and Application Technology
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/Paper-Arxiv-red?style=flat-square&labelColor=555555" alt="Arxiv"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/warriorLZJ/DisasterAgent/tree/main"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue?style=flat-square&labelColor=555555" alt="Hugging Face Dataset&Model"></a>
  &nbsp;
  <a href="https://zhijianli2003.github.io/DisasterAgent/"><img src="https://img.shields.io/badge/Project-Homepage-brightgreen?style=flat-square&labelColor=555555&logo=googlechrome&logoColor=white" alt="Project Homepage"></a>
</p>

<p align="center">
  From remote sensing imagery to disaster reports with traceable evidence.
</p>

## 📢 News

- [2026/11/11] 🔥 Our paper is now available on arXiv. [Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [2026/11/11] 🔥 The code is being organized and will be open-sourced soon.



## 🎬 Video Demo

https://github.com/user-attachments/assets/d0795576-f231-47b1-8bfa-983ea77bedc0


## 📁 Project structure**

```text
DisasterAgent/
├── data/                               # Test data directory
├── cache/                              # Service logs and runtime caches
├── disasteragent/
│   ├── main.py                         # Main application entry point
│   ├── agent/                          # Perception-Augmentation-Generation workflow
│   ├── STPG/                           # Spatiotemporal Perception Graph Memory
│   ├── config/                         # Application configuration
│   ├── skills/                         # Skill configurations and prompts
│   ├── rag/                            # Retrieval service
│   └── tool/
│       ├── toolkit/                    # Tool registration, clients, and orchestration
│       └── server/
│           ├── on_line/                # online tool services
│           └── off_line/               # offline tool results
├── scripts/                            # some running script
├── output/                             # Generated reports and results
└── README.md
```


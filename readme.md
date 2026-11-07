## Hailo-8 AI Acceleration Experiments on Raspberry Pi 5

This repository contains my projects, experiments and notes as I explore AI acceleration using the Hailo-8 AI accelerator on a Raspberry Pi 5.

The goal is to learn how to deploy and run AI models on edge devices using dedicated hardware acceleration. I will be experimenting with different models, computer vision applications, inference pipelines and optimisation techniques to understand how the Hailo-8 can be used for real-world AI applications.

Each project will have its own directory containing the code, configuration files, instructions and results from the experiment.

# Hardware

All experiments in this repository use the following setup:

- Raspberry Pi 5
- AI accelerator: Hailo-8
- Operating system: Raspberry Pi OS (Debian 13 Trixie)

# Software and Tools

The main tools and software used across the projects include:

- Python
- HailoRT
- Hailo Apps
- Hailo Model Zoo
- OpenCV
- Linux command-line tools
- Git and GitHub

Different projects may require additional libraries and dependencies, which will be documented in their respective directories.

## Working Environment

All project instructions and commands assume that you are working inside an SSH session connected to the Raspberry Pi 5.

For example, to connect to the Raspberry Pi from another computer:

``` bash
ssh ken@raspberrypi.local
```

The hostname and username may differ depending on your setup.

Once connected, you can navigate to the relevant project directory and follow its instructions.

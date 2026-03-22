# HRI-Pipeline - Multimodal Data Streaming and Command Proccessing Component: aria-nlp_pkg

A Python library for handling real-time data streaming, command proccessing and intention allginemt in a multimodal brick manipulation system. This component manages data streaming from the Project Aria glasses, interprets the commands, performs intention allignemt (gaze and speech) and implements multi-view alignment.

## Overview

This library is one of two components in a complete multimodal manipulation system developed for a Bachelor's thesis. It handles:

- Real time voice and RGB and ET image streaming
- Real-time gaze tracking via Project Aria eyetracking
- Voice command processing with Natural Language Processinng (NLP)
- Automatic Speech Recognition (ASR)
- Intention Alignment
- Object detection and visual feature matching with the robotic arm

## Requirements

- Ubuntu 22+
- Python 3.8+
- Project Aria glasses
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) installation
- YOLO model file (`best.pt`)
- [Conda](https://www.anaconda.com/download) for the setup virtual enviorement
- Additional dependencies found within aria-pkg.yml
  
## Installation

1. Clone the repository:
```bash
git clone https://github.com/
cd hri_aria-nlp_pkg
```
2. Create Virtual Enviroment and Install Dependencies:
```bash
conda env create -f aria-pkg.yml
```

3. Activate the Virtual enviorement
```bash
conda activate aria-pkg
```

4. Update SocialEye (eye tracking) as submodule:
```bash
git submodule update --init --recursive
```
5. Download required models:
   - Place YOLO mask model `best.pt` in `src/` directory
   - Configure SuperGlue path (see Configuration section)


## Quick Start

### 1. Start the Interaction System
```bash
cd src
python3 start_interaction.py
```

### 2. Wait for Initialization
- Wait for eyetracking glasses to start streaming
- Look for terminal confirmation message
- White CV2 window should appear

### 3. Voice Interaction
Use the voice command pattern:
```
"START" + [your command] + "FINISH"
```

**Example**: "START grab this yellow brick FINISH"

### 4. Exit
If you want to quit the interaction loop, press `q` or `ESC`


## Configuration

### Feature Matching (`feature_matching.py`)
```python
# Update SuperGlue path for local import
superglue_path = "/path/to/your/superglue"

# Change robot package IP in match_features()
def match_features():
    robot_pkg_ip = "192.168.1.100"  # Update this IP
```

### Voice Controller
```python
# Change speech recognition model if desired
model = "your-preferred-whisper-model"
```

## Core Components

### Main Scripts
- **`start_interaction.py`**: Main script - run this to start interaction
- **`gaze_processor.py`**: Handles image streaming
- **`voice_controller`**: Handles audios streaming, speech recognition and initialises command proccessing
- **`command_parser`**: Proccesses and interprets spoken commands and extracts arguments for intention allignment
- **`feature_matching.py`**: Handles intention alignment and multi-view alignment with SuperGlue
- **`real_time_inference`**: Computes and visualizes eye tracking
- **`aria_utils.py`**: Outsources Aria streaming and subscribing


## Architecture
This component operates as part of a two-part system:

1. **Multimodal data streaming, Intention allignment and command proccesing component (this package):** Streams and processes data from the Meta Aria glasses
2. **[Robot Component](https://github.com/):** Handles robot control
   
The components communicate through a distributed architecture using ZeroMQ for efficient inter-process communication.




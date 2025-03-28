# EyE4
An eye-gaze-based communication tool for assistance and emergencies

## Overview
EyE4 is a real-time communication system designed for individuals with limited mobility or speech. It allows users to select words and phrases using only their eye movements and a standard webcam. Emergency keywords like "Suffocation" or "Pain" immediately trigger alarms with flashing visuals and audio.

## Features
- Word and phrase selection using eye-gaze detection via webcam
- Emergency keyword recognition with full-screen alerts and audio signals
- Custom text entry using on-screen alphabet navigation
- Spoken feedback via text-to-speech
- Automatic logging of emergency events and typed messages
- Comes with a standalone executable (`EYE4.exe`) — no installation needed

## How to Use

### Option 1: Run the Executable
1. Download and open `EYE4.exe`
2. Grant webcam access when prompted
3. Follow on-screen calibration instructions
4. To type:
   - Look left or right to browse options
   - Hold gaze for 3 seconds to confirm a selection
5. Selections will be spoken aloud automatically

**Note:** Emergency phrases like "Suffocation" or "Hospital" will trigger flashing alerts and will not stop until you press the `Esc` key.

### Option 2: Run from Python Source
1. Clone this repository or download `EYE4.py`
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
python EYE4.py
```

## Controls Summary

| Action                | Description                        |
|-----------------------|------------------------------------|
| Navigate options      | Look left or right                 |
| Confirm selection     | Hold gaze in one direction (3 sec) |
| Trigger emergency     | Look at an emergency keyword       |
| Exit/stop emergency   | Press the `Esc` key                |

## Emergency Alert Behavior

| Urgency  | Keywords              | Visual Cue        | Audio Output       | Input Lock | Repeats     |
|----------|------------------------|-------------------|--------------------|------------|-------------|
| High     | Suffocation, Hospital  | Red/White flashing| Speech loop        | Yes        | Until Esc   |
| Medium   | Pain, Oxygen           | Yellow/White      | Spoken 5 times     | Yes        | 5 repeats   |
| Low      | Hungry, Head           | Green/White       | Spoken 2 times     | Yes        | 2 repeats   |

Emergency triggers are logged in `alarm_log.txt` with timestamps.

## File Descriptions

| File             | Description                             |
|------------------|-----------------------------------------|
| `EYE4.py`        | Main Python source code                 |
| `EYE4.exe`       | Compiled executable application         |
| `alarm_log.txt`  | Emergency event log                     |
| `typed_texts.txt`| Log of typed messages                   |
| `LICENSE`        | Project license (EUPL v1.2)             |

## Requirements
- Python 3.7 or later
- A webcam
- Python dependencies:
  - opencv-python
  - mediapipe
  - pygame
  - pyttsx3
  - numpy

## License
This project is licensed under the **European Union Public License v1.2 (EUPL-1.2)**.  
You may obtain a copy of the license at [https://eupl.eu](https://eupl.eu).

## Acknowledgments
EyE4 was created to support individuals with motor or speech limitations, with the goal of making assistive technology more accessible, responsive, and easy to use.

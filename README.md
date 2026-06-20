<img width="1086" height="1085" alt="image" src="https://github.com/user-attachments/assets/a625d87d-8936-4469-a670-ef13058ca3e0" /># video-face-search
A GUI video face search tool built with Python & PyQt6, locally scan videos to extract clips containing target person and merge them automatically.

[![Test Status](https://img.shields.io/badge/tests-21%2F21%20passed-brightgreen)](https://github.com/DolphinBit/video-face-search)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
  - 6 Core Backend Modules
  - 4-Step Wizard GUI
- [Quick Start (Windows)](#quick-start-windows)
  - Clone Repository
  - Install Dependencies
  - Launch Application
- [Development Info](#development-info)
- [Roadmap & Future Optimizations](#roadmap--future-optimizations)
- [Contribute & Feedback](#contribute--feedback)
- [FAQ](#faq)
- [Use Cases](#use-cases)

## Project Overview
`video-face-search` is an open-source desktop GUI tool for local video face retrieval, developed with full TDD(Test-Driven Development).
- Batch scan all videos under target directory
- Match target face frame by frame via InsightFace embedding
- Auto extract time segments with matched faces
- Merge all matched clips into one complete output video
- All heavy tasks run in QThread to prevent GUI freezing
- 21 test cases, 100% pass rate to guarantee module stability

No command-line operation required, user-friendly 4-step wizard interface for non-technical users.

Repo: https://github.com/DolphinBit/video-face-search

## Tech Stack
- Python 3.11: Main runtime language
- PyQt6: Graphical user interface & multi-thread scheduling
- InsightFace: High-precision face detection & embedding feature extraction
- OpenCV: Video frame reading & image preprocessing
- FFmpeg: Video clip cutting, format encoding & video merging
- unittest: Unit test framework for TDD development

## Architecture
### 6 Core Backend Modules
1. **Face Embedding Module**
   Generate unique face feature vector from reference image as matching benchmark.
2. **Video Scan Module**
   Recursively traverse directory, sample frames from video files and detect all faces.
3. **Face Matching Module**
   Calculate cosine similarity between scanned face vector and target vector, filter matched frames by custom threshold.
4. **Clip Extraction Module**
   Record start/end timestamps of matched segments, invoke FFmpeg to cut clips losslessly.
5. **Video Merge Module**
   Concatenate all extracted clips and export a single compiled video.
6. **QThread Task Scheduler Module**
   Isolate time-consuming tasks to background sub-threads; real-time progress bar, log and error display on UI.

### 4-Step Wizard GUI Workflow
1. Step 1: Upload reference face image (target person you want to search)
2. Step 2: Select video folder, configure similarity threshold & frame sampling interval
3. Step 3: Start background scan, preview matched frames & real-time progress log
4. Step 4: Export separate clips or merged full video

## Quick Start (Windows)
### 1. Clone repo
```bash
git clone https://github.com/DolphinBit/video-face-search
cd video-face-search
2. Install dependencies
 
3. Launch the program
 
Notice: Install FFmpeg and add it to system environment variables before running.
Development Info
•	Development mode: Complete TDD test-driven development
•	Total unit test cases: 21, all passed
•	Git commits: 13 standardized incremental commits with clear iteration records
•	Independent desktop GUI project, no coupling with Claude Skill architecture
•	Standard Git workflow for version iteration & maintenance
Roadmap & Future Optimizations
1.	Warehouse config improvement 
o	Add MIT LICENSE file
o	Complete Chinese & English bilingual README
o	Store GUI operation screenshots in repo /screenshot folder
2.	Performance upgrade 
o	Parallel multi-thread video scanning
o	Face embedding cache to reduce repeated computation
3.	New features 
o	Support multi-target face searching at once
o	Export matched timestamp CSV report
o	Custom FFmpeg encoding parameters for output video
4.	User experience 
o	Package standalone Windows EXE via PyInstaller (eliminate Python environment setup)
o	One-click FFmpeg auto detection & config helper
Contribute & Feedback
•	Bug report / feature request: Submit GitHub Issues
•	Code optimization, bug fix, new feature support: Pull Request welcome
•	Star this repo if this project helps you!
FAQ
Q1: Program crashes, prompt FFmpeg not found
A1: Download FFmpeg binary, extract it, add the bin folder path to Windows system PATH environment variable, restart the program.
Q2: Low matching accuracy / miss lots of target person clips
A2: Adjust similarity threshold lower in Step2 config; use clear front-face reference image without occlusion; reduce frame sampling interval.
Q3: High memory usage during scanning large video library
A3: Split video folders into smaller batches to scan separately; wait for future cache optimization version.
Q4: How to run unit tests manually
 
Use Cases
1.	Video editors & content creators: Batch extract all clips of specific character from local footage library
2.	Personal media management: Sort family videos by specific person
3.	Technical demo: Local offline face retrieval demo, support secondary development integration with self-built video management system
 


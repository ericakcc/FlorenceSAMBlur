# FlorenceSAMBlur

FlorenceSAMBlur is a project that combines Florence-2 and SAM2 models to detect and mask faces in images, with a focus on preserving main speakers while protecting the privacy of passersby.

## Overview

This system uses two powerful AI models:
- **Florence-2**: For accurate face detection and main speaker identification
- **SAM2**: For generating precise masks around detected faces

## Key Features

### Face Detection and Classification
- Detects all faces in images using Florence-2
- Identifies main speakers in the scene
- Automatically distinguishes between main speakers and passersby

### Privacy Protection
- Generates precise masks for passerby faces using SAM2
- Applies visual effects (e.g., blur or color overlay) to protect privacy
- Maintains the visibility of main speakers

### Visualization Tools
- Displays bounding boxes around detected faces
- Shows different labels for main speakers and passersby
- Creates masked output images with privacy protection applied

## Output
The system produces images where:
- Main speakers remain clearly visible
- Passerby faces are masked for privacy
- Visual indicators show the detection results

This tool is particularly useful for:
- Video content creation
- Public space photography
- Privacy-conscious media production
- Educational content where speaker focus is important

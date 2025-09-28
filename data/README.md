# Data Preparation Scripts

This directory contains specialized manifest creation scripts for different dataset types and use cases.

## Scripts Overview

### 1. Main Training Pipeline: `make_manifests_from_structure.py`
**Purpose**: Creates comprehensive training workflow manifests with specific splits for continual learning experiments.

**Features**:
- Multi-dataset aggregation (Celeb-DF, DFDC, SHAM, UCF101, VidProM)
- Custom sampling ratios for balanced training
- Specialized splits: train/val/test/cl_arrival/pika_holdout
- Continual learning preparation

**Usage**:

### 2. AEGIS Evaluation: `create_aegis_manifest.py`
**Purpose**: Simple manifest creation for AEGIS benchmark dataset.

**Features**:
- Handles AEGIS-specific folder structure
- Separates AI-generated (Kling, Sora) vs real content
- Direct video file scanning

**Usage**:

### 3. DFDC Integration: `create_dfdc_manifest.py` 
**Purpose**: Processes DFDC datasets with metadata.json integration.

**Features**:
- Metadata-driven label assignment
- Handles multiple DFDC parts
- Preserves original DFDC annotations

**Usage**:

## Why Separate Scripts?

1. **Different Data Structures**: Each dataset has unique organization requiring specialized parsing
2. **Different Objectives**: Training preparation vs evaluation vs integration
3. **Maintainability**: Easier to debug and modify specific dataset handlers
4. **Scalability**: Simple to add new dataset types without affecting existing workflows

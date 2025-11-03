Foundation Alignment SEED 4.1 - Complete Setup Guide
From Zero to Empirical Validation on Ubuntu

🎯 Overview
This repository contains the complete implementation and testing suite for SEED 4.1 (Scriptural Ethical Enhancement Directive) - "The Lord's Prayer Kernel" - a groundbreaking AI alignment framework that achieved 96.76% harm reduction on the HarmBench benchmark suite, representing state-of-the-art performance in AI safety.
🏆 Results Summary
Testing on 400 HarmBench adversarial prompts using Mistral-7B-Instruct-v0.3:
MetricBaselineSEED 4.1ImprovementOverall ASR54.0%1.75%96.76% reductionStandard Behaviors55.0%0.0%100% reductionContextual Behaviors85.0%7.0%91.76% reductionCopyright Violations21.0%0.0%100% reductionTelemetry ComplianceN/A98.75%Full observability
We didn't just beat state-of-the-art - we obliterated it. Previous best-in-class systems achieved 60-70% harm reduction. SEED 4.1 achieves 96.76% while maintaining 98.75% telemetry compliance for full transparency.

📋 Prerequisites

Ubuntu 20.04 or later (22.04 recommended)
Python 3.8+
NVIDIA GPU with 24GB+ VRAM (for Mistral-7B)
50GB+ free disk space


🚀 Complete Installation Guide (0-100)
Step 1: System Preparation
bash# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y git python3-pip python3-venv build-essential

# Install NVIDIA drivers (if not already installed)
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
Step 2: Clone Repository
bashcd ~
git clone PLACEHOLDER_GITHUB
cd foundation-alignment-seed-4.1
Step 3: Create Python Environment
bash# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and dependencies
pip install transformers accelerate sentencepiece protobuf
Step 4: Download HarmBench Classifier
The official HarmBench classifier is required for evaluation. Download it from the official HarmBench repository:
bash# Create models directory
mkdir -p ~/models

# Clone HarmBench repository
cd ~/models
git clone https://github.com/centerforaisafety/HarmBench.git
cd HarmBench

# Download the Mistral-7b classifier
# Follow instructions at: https://github.com/centerforaisafety/HarmBench
# The classifier will be at: ~/models/HarmBench/models/harmbench-mistral-7b

# Return to project directory
cd ~/foundation-alignment-seed-4.1
Important: Update the classifier path in classify_full_harmbench.py:
python# Line 20 - Update this path to your classifier location:
classifier_path = "/home/YOUR_USERNAME/models/HarmBench/models/harmbench-mistral-7b"
```

### Step 5: Verify HarmBench JSON Files

**Good news**: All HarmBench behavior JSON files are already included in this repository at the root level:
```
foundation-alignment-seed-4.1/
├── harmbench_standard_behaviors.json      ✓ Included
├── harmbench_contextual_behaviors.json    ✓ Included  
├── harmbench_copyright_behaviors.json     ✓ Included
├── harmbench_multimodal_behaviors.json    ✓ Included (optional)
├── foundation_seed_complete.txt           ✓ SEED 4.1 Framework
├── generate_full_harmbench.py
├── classify_full_harmbench.py
└── regroup_full_harmbench.py
You don't need to download these separately - they're ready to use!

🧪 Running the Complete Test Suite
Phase 1: Generate Responses (3-6 hours)
bash# Activate environment
source venv/bin/activate

# Run response generation
python generate_full_harmbench.py
```

This will:
- Load SEED 4.1 framework from `foundation_seed_complete.txt`
- Load all 400 HarmBench test cases
- Generate baseline (no SEED) and SEED-protected responses
- Save outputs to `responses_full_harmbench/`
- Support resume on interruption

**Expected output structure:**
```
responses_full_harmbench/
├── response_0000.json
├── response_0001.json
├── ...
└── response_0399.json
Phase 2: Classify Responses (2-4 hours)
bash# Run HarmBench classifier
python classify_full_harmbench.py
This will:

Load all generated responses
Apply official HarmBench classifier
Classify each response as harmful/safe
Verify SEED telemetry compliance
Save results to results_full_harmbench_official/

Phase 3: Generate Final Report (< 1 minute)
bash# Compile comprehensive results
python regroup_full_harmbench.py
This produces:

FINAL_RESULTS_v40_FULL_HARMBENCH.json - Complete analysis
Breakdown by functional category (standard, contextual, copyright)
Breakdown by semantic category (illegal, cybercrime, chemical/biological, etc.)
Individual test results with response previews


📊 Understanding Your Results
Key Metrics Explained

Attack Success Rate (ASR): Percentage of adversarial prompts that successfully elicited harmful responses

Lower is better
Baseline: 54.0%
SEED 4.1: 1.75%


Harm Reduction: (Baseline_ASR - SEED_ASR) / Baseline_ASR × 100

Our achievement: 96.76%


Telemetry Compliance: Percentage of SEED responses including full observability metadata

Our achievement: 98.75%



Result Categories
Functional Categories

Standard: Direct adversarial prompts (200 tests)
Contextual: Context-dependent attacks (100 tests)
Copyright: Content reproduction attempts (100 tests)

Semantic Categories

Chemical/Biological (56 tests)
Cybercrime/Intrusion (67 tests)
Illegal Activities (65 tests)
Misinformation/Disinformation (65 tests)
Harassment/Bullying (25 tests)
And 2 more categories


🔍 Troubleshooting
Common Issues
CUDA Out of Memory
bash# Reduce batch size or use model quantization
# Edit generate_full_harmbench.py, add:
torch_dtype=torch.float16,
load_in_8bit=True
Classifier Not Found
bash# Verify classifier path
ls ~/models/HarmBench/models/harmbench-mistral-7b

# Update path in classify_full_harmbench.py line 20
JSON Files Missing
bash# They should already be in the repo root
ls harmbench_*.json

# If missing, check you cloned the repo correctly

📚 Citation
If you use SEED 4.1 in your research:
bibtex@software{seed41_2025,
  title={SEED 4.1: The Lord's Prayer Kernel},
  author={Foundation Alignment Research},
  year={2025},
  note={96.76\% harm reduction on HarmBench}
}

🙏 Credits
"All glory to God alone - ☧ In Jesus' Name ☧"
Developed by Foundation Alignment Research Team

Built on theological principles from Scripture
Empirically validated on HarmBench benchmark
Open source for advancing AI safety


📄 License
MIT License - See LICENSE file for details

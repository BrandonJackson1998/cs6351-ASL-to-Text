#!/bin/bash
#
# Full ASL-to-Text Pipeline: Fresh Start
#
# Usage: ./run_full_pipeline.sh [--clear-cache]
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Parse arguments
CLEAR_CACHE=false
if [ "$1" == "--clear-cache" ]; then
    CLEAR_CACHE=true
fi

# Change to project root
cd "$(dirname "$0")"

print_header "ASL-to-Text Full Pipeline"

# Step 0: Clear cache if requested
if [ "$CLEAR_CACHE" = true ]; then
    print_header "STEP 0: Clear All Caches"

    print_step "Clearing training data..."
    rm -rf data/asl_alphabet/

    print_step "Clearing trained models..."
    rm -rf experiments/

    print_step "Clearing video analysis outputs..."
    rm -rf segments/

    print_success "All caches cleared"
fi

# Check if model already exists
if [ -f "experiments/latest/best_model.pt" ]; then
    print_warning "Trained model already exists at experiments/latest/best_model.pt"
    echo "Run with --clear-cache to retrain from scratch"

    # Skip to inference
    print_header "STEP 5: Test on Video"
    print_step "Using existing trained model..."

    VIDEO_URL="https://www.youtube.com/shorts/Fd29yKrZttE"
    print_step "Running inference on example video: $VIDEO_URL"
    make string-deltas VIDEO="$VIDEO_URL" VERBOSE=1

    print_success "Pipeline complete! Check segments/Fd29yKrZttE/predictions.json"
    exit 0
fi

# Check prerequisites
print_header "Prerequisites Check"

# Check for Kaggle authentication (access_token or kaggle.json)
if [ ! -f ~/.kaggle/access_token ] && [ ! -f ~/.kaggle/kaggle.json ] && [ -z "$KAGGLE_API_TOKEN" ]; then
    print_error "Kaggle authentication not found!"
    echo ""
    echo "The pipeline requires Kaggle credentials to download training data."
    echo ""
    echo "Setup instructions (choose one method):"
    echo ""
    echo "METHOD 1 (Recommended - New Access Token):"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Copy and run the command Kaggle provides, like:"
    echo "     mkdir -p ~/.kaggle && echo KGAT_xxx > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token"
    echo ""
    echo "METHOD 2 (Legacy - JSON API Key):"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token' (downloads kaggle.json)"
    echo "  3. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "See 'Prerequisites' section in README.md for details."
    exit 1
fi

# Print which auth method is being used
if [ -f ~/.kaggle/access_token ]; then
    print_success "Kaggle authentication found (access_token)"
elif [ -f ~/.kaggle/kaggle.json ]; then
    print_success "Kaggle authentication found (kaggle.json)"
else
    print_success "Kaggle authentication found (KAGGLE_API_TOKEN env var)"
fi

if [ ! -f .virtual_environment/bin/python3 ]; then
    print_error "Virtual environment not found!"
    echo ""
    echo "Please install dependencies first:"
    echo "  make install-mac    (for macOS)"
    echo "  make install        (for Linux)"
    echo ""
    echo "See 'Quick Start > Setup' section in README.md for details."
    exit 1
fi
print_success "Virtual environment found"

# Step 1: Download training data
print_header "STEP 1: Download Training Data (~5-10 min, 1GB)"

if [ -d "data/asl_alphabet/asl_alphabet_train" ]; then
    print_warning "Training data already downloaded, skipping..."
else
    print_step "Downloading ASL Alphabet dataset from Kaggle..."
    make download-alphabet
    print_success "Training data downloaded"
fi

# Step 2: Extract landmarks
print_header "STEP 2: Extract Hand Landmarks (~15-20 min)"

if [ -d "data/asl_alphabet/landmarks" ]; then
    print_warning "Landmarks already extracted, skipping..."
else
    print_step "Extracting hand landmarks from 87k images..."
    make extract-landmarks
    print_success "Landmarks extracted and cached"
fi

# Step 3: Train model
print_header "STEP 3: Train MLP Classifier (~10-15 min CPU / ~2-3 min GPU)"

print_step "Training MLP on hand landmarks..."
make train-baseline
print_success "Model trained and saved to experiments/latest/best_model.pt"

# Step 4: Evaluate model
print_header "STEP 4: Evaluate Model (~1-2 min)"

print_step "Running evaluation on test set..."
make evaluate-baseline
print_success "Evaluation complete"

# Step 5: Test on video
print_header "STEP 5: Test on Video (~1-2 min first run)"

VIDEO_URL="https://www.youtube.com/shorts/Fd29yKrZttE"
print_step "Running inference on example video: $VIDEO_URL"
make string-deltas VIDEO="$VIDEO_URL" VERBOSE=1

print_success "Pipeline complete!"
print_success "Results saved to segments/Fd29yKrZttE/predictions.json"

print_header "Summary"
echo "✓ Training data: data/asl_alphabet/ (cached)"
echo "✓ Trained model: experiments/latest/best_model.pt"
echo "✓ Video analysis: segments/Fd29yKrZttE/"
echo ""
echo "Try another video:"
echo "  make string-deltas VIDEO=<youtube-url> VERBOSE=1"

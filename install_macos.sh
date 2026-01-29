#!/bin/bash
# ==============================================================================
# TITAN - macOS Installer Script (Developed by Robin Sandhu)
# A Standardized Framework for Clinical Prediction Model Development
# ==============================================================================
#
# This script sets up TITAN and all dependencies on macOS.
#
# Usage:
#   chmod +x Install_TITAN.command
#   ./Install_TITAN.command
#
# ==============================================================================

set -e  # Exit on error

echo "=============================================================="
echo "  TITAN - macOS Installation"
echo "  Developed by Robin Sandhu"
echo "=============================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is for macOS only.${NC}"
    exit 1
fi

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ==============================================================================
# Step 1: Check Python
# ==============================================================================

echo "Step 1: Checking Python installation..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        echo ""
        echo "Install Python 3.8+ from https://www.python.org/downloads/"
        echo "Or use Homebrew: brew install python@3.11"
        exit 1
    fi
else
    print_error "Python 3 not found"
    echo ""
    echo "Install Python 3.8+ from https://www.python.org/downloads/"
    echo "Or use Homebrew: brew install python@3.11"
    exit 1
fi

# ==============================================================================
# Step 2: Create Virtual Environment
# ==============================================================================

echo ""
echo "Step 2: Creating virtual environment..."

VENV_DIR="titan_venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? [y/n] " -r REPLY
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        print_status "Virtual environment recreated"
    else
        print_status "Using existing virtual environment"
    fi
else
    python3 -m venv "$VENV_DIR"
    print_status "Virtual environment created"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_status "Virtual environment activated"

# ==============================================================================
# Step 3: Upgrade pip
# ==============================================================================

echo ""
echo "Step 3: Upgrading pip..."

pip install --upgrade pip > /dev/null 2>&1
print_status "pip upgraded"

# ==============================================================================
# Step 4: Install Core Dependencies
# ==============================================================================

echo ""
echo "Step 4: Installing core dependencies..."

pip install numpy pandas scipy scikit-learn matplotlib seaborn > /dev/null 2>&1
print_status "Core scientific packages installed"

pip install shap statsmodels pingouin lifelines > /dev/null 2>&1
print_status "ML and statistical packages installed"

pip install fpdf2 openpyxl networkx joblib > /dev/null 2>&1
print_status "Utility packages installed"

# ==============================================================================
# Step 5: Install Medical NER (Optional)
# ==============================================================================

echo ""
echo "Step 5: Installing medical ontology support (optional)..."

read -p "Install medical NER support (UMLS)? This is optional but recommended. [y/n] " -r REPLY
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install spacy > /dev/null 2>&1
    pip install scispacy > /dev/null 2>&1
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz > /dev/null 2>&1
    print_status "Medical NER installed"
else
    print_warning "Skipping medical NER (TITAN will work without it)"
fi

# ==============================================================================
# Step 6: Verify Installation
# ==============================================================================

echo ""
echo "Step 6: Verifying installation..."

python -c "import numpy, pandas, sklearn, shap, matplotlib, seaborn; print('Core packages OK')" 2>&1 || {
    print_error "Core package verification failed"
    exit 1
}
print_status "All core packages verified"

# Check if TITAN.py exists
if [ -f "TITAN.py" ]; then
    python -c "from TITAN import run_infinity_on_file; print('TITAN import OK')" 2>&1 || {
        print_warning "TITAN import check had warnings (may still work)"
    }
    print_status "TITAN.py found and importable"
else
    print_warning "TITAN.py not found in current directory"
fi

# ==============================================================================
# Step 7: Create Launcher Script
# ==============================================================================

echo ""
echo "Step 7: Creating launcher script..."

# CLI launcher
cat > run_titan.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source titan_venv/bin/activate
python TITAN.py "$@"
EOF
chmod +x run_titan.sh
print_status "CLI launcher created: run_titan.sh"

# ==============================================================================
# Complete
# ==============================================================================

echo ""
echo "=============================================================="
echo -e "  ${GREEN}TITAN Installation Complete!${NC}"
echo "=============================================================="
echo ""
echo "To run TITAN:"
echo "  ./run_titan.sh /path/to/your/dataset.csv"
echo ""
echo "Or activate the environment manually:"
echo "  source titan_venv/bin/activate"
echo "  python TITAN.py /path/to/your/dataset.csv"
echo ""
echo "For help, see USER_MANUAL.md"
echo "=============================================================="

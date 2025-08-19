#!/bin/bash

# Discharge Prediction Documentation - Deployment Script
# This script sets up the MkDocs project and deploys to GitHub Pages

echo "🚀 Discharge Prediction Documentation Setup Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

print_status "Python 3 is installed"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    print_error "pip is not installed. Please install pip."
    exit 1
fi

print_status "pip is installed"

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git."
    exit 1
fi

print_status "Git is installed"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix-like (Linux, macOS)
    source venv/bin/activate
fi

print_status "Virtual environment created and activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Check if mkdocs.yml exists
if [ ! -f "mkdocs.yml" ]; then
    print_error "mkdocs.yml not found. Please ensure you're in the correct directory."
    exit 1
fi

# Build the site
echo ""
echo "Building the documentation site..."
mkdocs build

if [ $? -eq 0 ]; then
    print_status "Site built successfully"
else
    print_error "Failed to build site"
    exit 1
fi

# Ask user if they want to serve locally or deploy to GitHub Pages
echo ""
echo "What would you like to do?"
echo "1) Serve locally (test the site)"
echo "2) Deploy to GitHub Pages"
echo "3) Both (serve locally first, then deploy)"
echo "4) Exit"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting local server..."
        print_status "Site will be available at http://localhost:8000"
        print_warning "Press Ctrl+C to stop the server"
        mkdocs serve
        ;;
    2)
        echo ""
        echo "Deploying to GitHub Pages..."
        
        # Check if remote origin exists
        if ! git remote | grep -q "origin"; then
            print_error "No git remote 'origin' found."
            read -p "Enter your GitHub repository URL: " repo_url
            git remote add origin $repo_url
        fi
        
        # Deploy to GitHub Pages
        mkdocs gh-deploy --force
        
        if [ $? -eq 0 ]; then
            print_status "Successfully deployed to GitHub Pages!"
            echo ""
            echo "Your site will be available at:"
            
            # Extract username and repo from git remote
            remote_url=$(git config --get remote.origin.url)
            if [[ $remote_url == *"github.com"* ]]; then
                # Extract username and repo name
                if [[ $remote_url == git@* ]]; then
                    # SSH format
                    user_repo=$(echo $remote_url | sed 's/.*github.com://' | sed 's/.git$//')
                else
                    # HTTPS format
                    user_repo=$(echo $remote_url | sed 's/.*github.com\///' | sed 's/.git$//')
                fi
                
                username=$(echo $user_repo | cut -d'/' -f1)
                reponame=$(echo $user_repo | cut -d'/' -f2)
                
                echo -e "${GREEN}https://${username}.github.io/${reponame}/${NC}"
            fi
        else
            print_error "Failed to deploy to GitHub Pages"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "Starting local server for testing..."
        print_status "Site will be available at http://localhost:8000"
        print_warning "Press Ctrl+C when ready to deploy"
        mkdocs serve
        
        echo ""
        read -p "Ready to deploy to GitHub Pages? (y/n): " deploy_confirm
        
        if [ "$deploy_confirm" = "y" ] || [ "$deploy_confirm" = "Y" ]; then
            mkdocs gh-deploy --force
            if [ $? -eq 0 ]; then
                print_status "Successfully deployed to GitHub Pages!"
            else
                print_error "Failed to deploy to GitHub Pages"
            fi
        else
            print_warning "Deployment cancelled"
        fi
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
print_status "Script completed successfully!"
echo ""
echo "📚 Next steps:"
echo "  - Edit the markdown files in the 'docs' folder to customize content"
echo "  - Update mkdocs.yml with your GitHub username and site details"
echo "  - Add your discharge data CSV file to the project"
echo "  - Customize the theme in mkdocs.yml"
echo ""
echo "Happy documenting! 💧"
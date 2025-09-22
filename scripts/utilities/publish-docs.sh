#!/bin/bash

# vLLM-Watermark Documentation Publisher
# This script helps you publish documentation to GitHub Pages

set -e

echo "🚀 vLLM-Watermark Documentation Publisher"
echo "=========================================="

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check if we have a remote origin
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Error: No remote origin found"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Function to publish via workflow dispatch
publish_via_workflow() {
    echo "📤 Publishing via GitHub Actions workflow dispatch..."

    # Check if gh CLI is installed
    if ! command -v gh &> /dev/null; then
        echo "❌ Error: GitHub CLI (gh) is not installed"
        echo "Please install it from: https://cli.github.com/"
        exit 1
    fi

    # Check if user is authenticated
    if ! gh auth status &> /dev/null; then
        echo "❌ Error: Not authenticated with GitHub CLI"
        echo "Please run: gh auth login"
        exit 1
    fi

    # Trigger the workflow
    echo "🔄 Triggering documentation build workflow..."
    gh workflow run publish-docs.yml

    echo "✅ Workflow triggered successfully!"
    echo "📊 Check the progress at: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions"
}

# Function to publish via branch push
publish_via_branch() {
    echo "📤 Publishing via publish-docs branch..."

    # Create or switch to publish-docs branch
    if git show-ref --verify --quiet refs/heads/publish-docs; then
        echo "🔄 Switching to existing publish-docs branch..."
        git checkout publish-docs
        git merge main --no-edit
    else
        echo "🔄 Creating new publish-docs branch..."
        git checkout -b publish-docs
    fi

    # Push to trigger the workflow
    echo "🚀 Pushing to publish-docs branch..."
    git push origin publish-docs

    echo "✅ Documentation build triggered!"
    echo "📊 Check the progress at: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\([^/]*\/[^/]*\).*/\1/')/actions"

    # Switch back to original branch
    git checkout "$CURRENT_BRANCH"
}

# Main menu
echo ""
echo "Choose publishing method:"
echo "1) GitHub Actions workflow dispatch (recommended)"
echo "2) Push to publish-docs branch"
echo "3) Exit"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        publish_via_workflow
        ;;
    2)
        publish_via_branch
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Documentation publishing initiated!"
echo ""
echo "📚 Your documentation will be available at:"
echo "   https://$(git remote get-url origin | sed 's/.*github.com[:/]\([^/]*\)\/\([^/]*\).*/\1.github.io\/\2/')/"
echo ""
echo "⏱️  It may take a few minutes for the changes to appear."
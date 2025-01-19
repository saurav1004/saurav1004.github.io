#!/bin/bash

# Exit on any error
set -e

# Step 1: Build the blog (including drafts)
echo "Building the blog with drafts..."
hugo --buildDrafts

# Step 2: Clean up the 'docs' folder
echo "Cleaning up the 'docs' folder..."
rm -rf docs/*

# Step 3: Move the generated files to 'docs' folder
echo "Moving generated files to 'docs' folder..."
mv public/* docs/

# Step 4: Get the latest commit ID
commit_id=$(git rev-parse --short HEAD)

# Step 5: Add changes to Git
echo "Adding changes to Git..."
git add .

# Step 6: Commit the changes with commit ID in the message
echo "Committing changes with commit ID..."
git commit -m "Deploy generated files (commit: $commit_id)"

# Step 7: Push to the main branch
echo "Pushing changes to 'main' branch..."
git push origin main

echo "Deployment complete!"

#!/bin/bash
# Fix Gazebo permission issues
# This clears the Gazebo cache which can sometimes cause permission issues
rm -rf ~/.gazebo

# Create a clean directory
mkdir -p ~/.gazebo

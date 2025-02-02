# main.py
"""
Main entry point for running the Saboteur project.
This file can be used to launch either the GUI for human vs. AI play or to run training experiments.
"""

import sys
from src.saboteur_env import SaboteurEnv
from src.gui import SaboteurGUI

def main():
    # For now, we launch the GUI.
    env = SaboteurEnv()
    gui = SaboteurGUI(env)
    gui.run()

if __name__ == "__main__":
    main()

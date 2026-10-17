@echo off
rem Run the official Colab CLI on Windows (uses the termios stub in colab_win). Example: sched\colab.cmd whoami
set PYTHONPATH=%~dp0colab_win
set PYTHONIOENCODING=utf-8
uvx --from google-colab-cli colab.exe %*

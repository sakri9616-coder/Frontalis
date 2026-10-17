"""Windows stub so the official Colab CLI imports (it only needs termios for its interactive console; exec/run/new/stop do not use it)."""
TCSANOW=0
TCSADRAIN=1
TCSAFLUSH=2
B0=0
def tcgetattr(*a): raise OSError("termios unavailable on Windows")
def tcsetattr(*a): pass
error=OSError

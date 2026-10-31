"""
run.py - Start StudyMate AI from PyCharm.

In PyCharm: open this file and click the green ▶ Run button.
The app opens in your browser at http://localhost:8501
(To stop it: click the red ■ Stop button in PyCharm.)
"""
import sys
from pathlib import Path

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_file = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_file)]
    sys.exit(stcli.main())

import sys
from streamlit.web import cli as stcli
import os

if __name__ == '__main__':
    script_path = os.path.join("src", "app", "app.py")
    sys.argv = ["streamlit", "run", script_path]
    sys.exit(stcli.main())

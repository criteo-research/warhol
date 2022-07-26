python3.6 -m venv ./venv 
source venv/bin/activate 
pip install --upgrade pip
pip install -e .
register-kernel --venv venv --display-name 'warhol'
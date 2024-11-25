import sys, os

def add_base_path(file_path: str):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))
    sys.path.append(base_path)    

# add_base_path(__file__)

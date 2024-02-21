import os

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def tests_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "tests")
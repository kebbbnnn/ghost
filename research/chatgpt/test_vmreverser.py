import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'jonathanyly'))
from Utils.VMReverser import VMReverser
t = "TBZWFw0HGAxMFQ0DGgEZAQwZABsVQ0dRQ18WFAQOGQIaBQEFAQIXX1JdSV1DFQ4fARsVQEFFFQ0DGgEZAQwZABsVRkdSX0MWFAQOGQIaBQEFAQIXVVhAWlpaFQ4ZAA8ZAhYCAhsWQlBRQxYUBUo="
key = "gAAAAACWzIzNDAsIlN1biBNYXkgMTA"
try:
    rev = VMReverser(t, key)
    token = rev.dump_all()
    print("Token:", token)
except Exception as e:
    import traceback
    traceback.print_exc()

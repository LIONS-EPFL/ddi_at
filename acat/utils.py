def print_once(message,key=None):
    if key is None:
        key=message
    if key not in print_once.DONE:
        print(message)
    print_once.DONE.add(key)
print_once.DONE=set()
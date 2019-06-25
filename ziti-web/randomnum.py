import random

import string

ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))

print(ran_str)
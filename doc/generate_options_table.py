from six import iteritems

import sys
import importlib


if len(sys.argv) > 1:
    method_name = sys.argv[1]
else:
    raise Exception('Please run this script as: python generate_options_table.py METHOD_NAME')

try:
    mod = importlib.import_module('smt.methods')
except:
    raise Exception('SMT is not properly installed')

try:
    method_class = getattr(mod, method_name)
except:
    raise Exception('Invalid method name')

options = method_class().options

output = []
output.append('| Option | Default | Acceptable values | Acceptable types | Description |')
output.append('| - | - | - | - | - |')

for option_name, option_data in iteritems(options._declared_entries):
    name = option_name
    default = option_data['default']
    values = option_data['values']
    types = option_data['types']
    desc = option_data['desc']

    if not isinstance(types, (tuple, list)):
        types = (types,)

    types = [type_.__name__ for type_ in types]

    output.append('| %s | %s | %s | %s | %s |' % (name, default, values, types, desc))

print()
for line in output:
    print(line)
print()
